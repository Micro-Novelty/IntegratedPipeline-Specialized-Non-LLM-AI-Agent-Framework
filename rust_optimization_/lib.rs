// src/lib.rs
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use rusqlite::{Connection, params};
use serde_json::{Value, Map};
 /// Result type for array parsing: either a Vec<f32> or the original string.
use pyo3::types::PyList;
use std::str::FromStr;

/// Internal result type for the pure-Rust parsing logic.
#[derive(Debug, PartialEq)]
enum ParseResult {
    Array(Vec<f32>),
    Original(String),
}
 
/// Parse a string representation of an array into a Vec<f32>.
/// Returns `ParseResult::Array` on success, or `ParseResult::Original` if nothing worked.
fn parse_array_string_inner(s: &str) -> ParseResult {
    // Normalize whitespace (collapse \n, \r, \t, and multiple spaces)
    let s: String = s
        .chars()
        .map(|c| if c.is_ascii_whitespace() { ' ' } else { c })
        .collect();
    let s = s.split_whitespace().collect::<Vec<_>>().join(" ");
    let s = s.trim().to_string();
 
    if s.is_empty() {
        return ParseResult::Original(s);
    }
 
    // --- 1. JSON / bracket form ---
    if s.starts_with('[') && s.ends_with(']') {
        // Try serde_json first
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&s) {
            if let Some(arr) = parsed.as_array() {
                let floats: Option<Vec<f32>> = arr
                    .iter()
                    .map(|v| v.as_f64().map(|f| f as f32))
                    .collect();
                if let Some(floats) = floats {
                    return ParseResult::Array(floats);
                }
            }
        }
 
        // Fallback: strip brackets, split on commas or whitespace
        let inner = &s[1..s.len() - 1];
        if let Some(result) = parse_delimited(inner) {
            return ParseResult::Array(result);
        }
    }
 
    // --- 2. Looks like space/comma-separated numbers (possibly with brackets) ---
    // Equivalent to the Python regex r'[\[\]\s\d\.\,\-\+E]+'
    if s.chars().all(|c| "[]., \t\r\n+-eE0123456789".contains(c)) {
        let cleaned = s.replace(['[', ']'], " ");
        let parts: Vec<&str> = cleaned.split_whitespace().collect();
        if !parts.is_empty() {
            let floats: Result<Vec<f32>, _> = parts.iter().map(|p| f32::from_str(p)).collect();
            if let Ok(floats) = floats {
                return ParseResult::Array(floats);
            }
        }
    }
 
    // --- 3. Comma-separated values (brackets optional) ---
    if s.contains(',') {
        let cleaned = s.replace(['[', ']'], "");
        if let Some(result) = parse_delimited(&cleaned) {
            return ParseResult::Array(result);
        }
    }
 
    // Nothing worked — return original string
    ParseResult::Original(s)
}
 
/// Try to parse a string of comma- or whitespace-separated floats.
fn parse_delimited(s: &str) -> Option<Vec<f32>> {
    let parts: Vec<&str> = if s.contains(',') {
        s.split(',').map(str::trim).filter(|p| !p.is_empty()).collect()
    } else {
        s.split_whitespace().collect()
    };
 
    if parts.is_empty() {
        return None;
    }
 
    parts.iter().map(|p| f32::from_str(p).ok()).collect()
}
 
/// Parse a string representation of an array into a Python list of floats.
/// Returns the original string if parsing fails, mirroring the Python original.
///
/// Python usage:
///   from my_module import parse_array_string
///   result = parse_array_string("[1.0, 2.0, 3.0]")  # -> [1.0, 2.0, 3.0]
///   result = parse_array_string("not an array")      # -> "not an array"
#[pyfunction]
pub fn parse_array_string<'py>(py: Python<'py>, s: &str) -> PyResult<Bound<'py, PyAny>> {
    match parse_array_string_inner(s) {
        ParseResult::Array(floats) => {
            let list = PyList::new(py, &floats)?;
            Ok(list.into_any())
        }
        ParseResult::Original(original) => {
            Ok(original.to_object(py).into_ref(py))
        }
    }
}

#[pyfunction]
fn load_and_validate_model_dict(
    db_path: String,
    memory_name: String,
    expected_num_classes: usize,   // pass from _get_num_classes()
) -> PyResult<Option<String>> {
    // load from SQLite (already fast via rusqlite)
    let conn = Connection::open(&db_path)
        .map_err(|e| PyRuntimeError::new_err(format!("[!] DB open failed: {}", e)))?;

    let result: rusqlite::Result<String> = conn.query_row(
        "SELECT model_data FROM model_storage
         WHERE memory_name = ?1 AND is_active = 1
         ORDER BY id DESC LIMIT 1",
        params![memory_name],
        |row| row.get(0),
    );

    let json_str = match result {
        Ok(s) => s,
        Err(rusqlite::Error::QueryReturnedNoRows) => return Ok(None),
        Err(e) => return Err(PyRuntimeError::new_err(format!("[!] Query failed: {}", e))),
    };

    // parse and validate schema
    // serde_json is far faster than Python's json.loads for large blobs
    let mut data: Map<String, Value> = match serde_json::from_str(&json_str) {
        Ok(Value::Object(m)) => m,
        Ok(_) => return Ok(Some("{}".to_string())),  // not a dict — return empty
        Err(e) => return Err(PyRuntimeError::new_err(format!("[!] JSON parse failed: {}", e))),
    };

    // schema repair in Rust — remove corrupted entries
    let mut to_remove = Vec::new();

    for (key, value) in data.iter() {
        match value {
            Value::Null => {
                to_remove.push(key.clone());
            }
            Value::Array(arr) => {
                // check ndim > 2 equivalent — nested array depth
                let max_depth = arr.iter()
                    .filter_map(|v| v.as_array())
                    .map(|inner| inner.iter()
                         .filter_map(|v| v.as_array())
                         .count())
                    .max()
                    .unwrap_or(0);

                if max_depth > 0 {
                    // ndim > 2 — corrupted
                    to_remove.push(key.clone());
                    continue;
                }

                // check suspicious length against expected num_classes
                if expected_num_classes > 0 &&
                   arr.len() != expected_num_classes &&
                   arr.len() != expected_num_classes * 2 {
                    to_remove.push(key.clone());
                }
            }
            _ => {}  // strings, numbers, boolss
        }
    }

    for key in to_remove {
        data.remove(&key);
    }

    // return cleaned JSON string back to Python
    let cleaned = serde_json::to_string(&data)
        .map_err(|e| PyRuntimeError::new_err(format!("[!] JSON serialize failed: {}", e)))?;

    Ok(Some(cleaned))
}


#[pyfunction]
fn save_lstm_weights(
    db_path: String,
    memory_name: String,
    weights_json: String,
) -> PyResult<()> {
    let conn: Connection = Connection::open(&db_path)
        .map_err(|e: rusqlite::Error| PyRuntimeError::new_err(format!("[!] DB open failed: {}", e)))?;

    conn.execute(
        "INSERT INTO weight_storage (memory_name, model_type, weights, is_active)
         VALUES (?1, 'lstm', ?2, 1)",
        params![memory_name, weights_json],
    ).map_err(|e: rusqlite::Error| PyRuntimeError::new_err(format!("[!] Insert failed: {}", e)))?;

    conn.execute(
        "UPDATE weight_storage SET is_active = 0
         WHERE memory_name = ?1 AND model_type = 'lstm'
         AND id != last_insert_rowid()",
        params![memory_name],
    ).map_err(|e: rusqlite::Error| PyRuntimeError::new_err(format!("[!] Database Update failed: {}", e)))?;

    conn.execute(
        "DELETE FROM weight_storage
         WHERE memory_name = ?1 AND model_type = 'lstm' AND is_active = 0",
        params![memory_name],
    ).map_err(|e: rusqlite::Error| PyRuntimeError::new_err(format!("[!] Database Cleanup failed: {}", e)))?;

    Ok(())
}


#[pyfunction]
fn save_pipelines_dict(
    db_path: String, 
    memory_name: String,
    model_type: String,
    dict_json: String) -> PyResult<()> {
    let conn: Connection = Connection::open(&db_path)
        .map_err(|e: rusqlite::Error| PyRuntimeError::new_err(format!("[-] Failed to open database: {}", e)))?;

    if model_type == "Transformer" {
        conn.execute(
            "INSERT INTO model_attn_storage 
            (memory_name, model_type, model_data, is_active)
            VALUES (?1, ?2, ?3, 1)" ,
            params![memory_name, model_type, dict_json],      
        ).map_err(|e: rusqlite::Error| PyRuntimeError::new_err(format!("[-] Insert failed: {}", e)))?;
        

        conn.execute(
            "UPDATE model_attn_storage SET is_active = 0
            WHERE memory_name = ?1
            AND id != last_insert_rowid()",
            params![memory_name],
        ).map_err(|e: rusqlite::Error| PyRuntimeError::new_err(format!("[=] Update failed: {}", e)))?; 

        conn.execute(
            "DELETE FROM model_attn_storage
            WHERE memory_name = ?1 AND is_active = 0",
            params![memory_name],
        ).map_err(|e: rusqlite::Error| PyRuntimeError::new_err(format!("[=] Cleanup failed: {}", e)))?;
    }

    else {
        conn.execute(
            "INSERT INTO model_storage 
            (memory_name, model_type, model_data, is_active)
            VALUES (?1, ?2, ?3, 1)" ,
            params![memory_name, model_type, dict_json],      
        ).map_err(|e: rusqlite::Error| PyRuntimeError::new_err(format!("[-] Insert failed: {}", e)))?;
        

        conn.execute(
            "UPDATE model_storage SET is_active = 0
            WHERE memory_name = ?1
            AND id != last_insert_rowid()",
            params![memory_name],
        ).map_err(|e: rusqlite::Error| PyRuntimeError::new_err(format!("[=] Update failed: {}", e)))?; 

        conn.execute(
            "DELETE FROM model_storage
            WHERE memory_name = ?1 AND is_active = 0",
            params![memory_name],
        ).map_err(|e: rusqlite::Error| PyRuntimeError::new_err(format!("[=] Cleanup failed: {}", e)))?;        
    }
    Ok(())

}

#[pyfunction]
fn verify_memory_exist(
    db_path: String,
    memory_name: String,
    model_type: String,
) -> PyResult<bool> {  // Return PyResult<bool> — PyO3 auto-converts Rust bool to Python bool
    let conn: Connection = Connection::open(&db_path)
        .map_err(|e| PyRuntimeError::new_err(format!("[!] Failed to open database: {}", e)))?;

    // Select the correct table based on model_type
    let table = match model_type.as_str() {
        "Transformer" => "model_attn_storage",
        "Peer"        => "agent_attn_storage",
        _             => "model_storage",  // default / else branch
    };

    // Build query dynamically — LIMIT is not assigned with `=` in SQL
    let query = format!(
        "SELECT 1 FROM {} WHERE memory_name = ?1 AND is_active = 1 LIMIT 1",
        table
    );

    let result: rusqlite::Result<i32> = conn.query_row(  // SELECT 1 returns integer, not String
        &query,
        params![memory_name],
        |row| row.get(0),
    );

    // Map Ok → true (row found), NoRows error → false, other errors → propagate
    match result {
        Ok(_)                                           => Ok(true),
        Err(rusqlite::Error::QueryReturnedNoRows)       => Ok(false),
        Err(e)                                          => Err(PyRuntimeError::new_err(
                                                            format!("[!] Query failed: {}", e)
                                                          )),
    }
}

#[pyfunction]
fn load_attention_dict(db_path: String, memory_name: String) -> PyResult<Option<String>> {
    let conn: Connection = Connection::open(&db_path)
        .map_err(|e: rusqlite::Error| PyRuntimeError::new_err(format!("[-] Failed to open database: {}", e)))?;

  
    let result: rusqlite::Result<String> = conn.query_row(
        "SELECT model_data FROM model_attn_storage
         WHERE memory_name = ?1 AND is_active = 1
         ORDER BY id DESC LIMIT 1",
        params![memory_name],
        |row: &rusqlite::Row<'_>| row.get(0),
    );

    match result {
        Ok(json) => Ok(Some(json)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(PyRuntimeError::new_err(format!("[-] Query failed to load: {}", e))),
    }
}

#[pyfunction]
fn load_agent_id(
    db_path: String,
    memory_name: String
) -> PyResult<Option<String>> {
    let conn: Connection = Connection::open(&db_path)
            .map_err(|e: rusqlite::Error| PyRuntimeError::new_err(format!("[-] Failed to open database: {}", e)))?;    

    let result: rusqlite::Result<String> = conn.query_row(
        "SELECT agent_id FROM agent_attn_storage
        WHERE memory_name = ?1 AND is_active = 1",
        params![memory_name], |row: &rusqlite::Row<'_>| row.get(0),
    );

    match result {
        Ok(json) => Ok(Some(json)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(PyRuntimeError::new_err(format!("[!] Query failed: {}", e))),
    }
    
}

#[pyfunction]
fn load_lstm_weights(db_path: String, memory_name: String) -> PyResult<Option<String>> {
    let conn: Connection = Connection::open(&db_path)
        .map_err(|e: rusqlite::Error| PyRuntimeError::new_err(format!("[!] Database open failed: {}", e)))?;

    let result: rusqlite::Result<String> = conn.query_row(
        "SELECT weights FROM weight_storage
         WHERE memory_name = ?1 AND model_type = 'lstm' AND is_active = 1
         ORDER BY id DESC LIMIT 1",
        params![memory_name],
        |row: &rusqlite::Row<'_>| row.get(0),
    );

    match result {
        Ok(json) => Ok(Some(json)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(PyRuntimeError::new_err(format!("[!] Query failed: {}", e))),
    }
}

#[pyfunction]
fn save_transformer_weights(
    db_path: String,
    memory_name: String,
    binary_data: Vec<u8>,
) -> PyResult<()> {
    let conn: Connection = Connection::open(&db_path)
        .map_err(|e: rusqlite::Error| PyRuntimeError::new_err(format!("[!] Database open failed: {}", e)))?;

    conn.execute(
        "INSERT INTO weight_storage (memory_name, model_type, weights, is_active)
         VALUES (?1, 'transformer', ?2, 1)",
        params![memory_name, binary_data],
    ).map_err(|e: rusqlite::Error| PyRuntimeError::new_err(format!("[!] Database Insert failed: {}", e)))?;

    conn.execute(
        "UPDATE weight_storage SET is_active = 0
         WHERE memory_name = ?1 AND model_type = 'transformer'
         AND id != last_insert_rowid()",
        params![memory_name],
    ).map_err(|e: rusqlite::Error| PyRuntimeError::new_err(format!("[!] Update failed: {}", e)))?;

    conn.execute(
        "DELETE FROM weight_storage
         WHERE memory_name = ?1 AND model_type = 'transformer' AND is_active = 0",
        params![memory_name],
    ).map_err(|e: rusqlite::Error| PyRuntimeError::new_err(format!("[!] Cleanup failed: {}", e)))?;

    Ok(())
}

#[pyfunction]
fn load_transformer_weights(db_path: String, memory_name: String) -> PyResult<Option<Vec<u8>>> {
    let conn: Connection = Connection::open(&db_path)
        .map_err(|e: rusqlite::Error| PyRuntimeError::new_err(format!("[!] Database open failed: {}", e)))?;

    let result: rusqlite::Result<Vec<u8>> = conn.query_row(
        "SELECT weights FROM weight_storage
         WHERE memory_name = ?1 AND model_type = 'transformer' AND is_active = 1
         ORDER BY id DESC LIMIT 1",
        params![memory_name],
        |row: &rusqlite::Row<'_>| row.get(0),
    );

    match result {
        Ok(blob) => Ok(Some(blob)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(PyRuntimeError::new_err(format!("[!] Query failed: {}", e))),
    }
}

#[pymodule]
fn abstract_weights_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(save_lstm_weights, m)?)?;
    m.add_function(wrap_pyfunction!(load_lstm_weights, m)?)?;
    m.add_function(wrap_pyfunction!(save_transformer_weights, m)?)?;
    m.add_function(wrap_pyfunction!(load_transformer_weights, m)?)?;
    m.add_function(wrap_pyfunction!(load_attention_dict, m)?)?;
    m.add_function(wrap_pyfunction!(save_pipelines_dict, m)?)?;
    m.add_function(wrap_pyfunction!(load_and_validate_model_dict, m)?)?;
    m.add_function(wrap_pyfunction!(parse_array_string, m)?)?;
    m.add_function(wrap_pyfunction!(load_agent_id, m)?)?;
    m.add_function(wrap_pyfunction!(verify_memory_exist, m)?)?;
    Ok(())
}
