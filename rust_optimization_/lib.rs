// src/lib.rs
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use rusqlite::{Connection, params};
use serde_json::{Value, Map};


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
        .map_err(|e: rusqlite::Error| PyRuntimeError::new_err(format!("DB open failed: {}", e)))?;

    conn.execute(
        "INSERT INTO weight_storage (memory_name, model_type, weights, is_active)
         VALUES (?1, 'lstm', ?2, 1)",
        params![memory_name, weights_json],
    ).map_err(|e: rusqlite::Error| PyRuntimeError::new_err(format!("Insert failed: {}", e)))?;

    conn.execute(
        "UPDATE weight_storage SET is_active = 0
         WHERE memory_name = ?1 AND model_type = 'lstm'
         AND id != last_insert_rowid()",
        params![memory_name],
    ).map_err(|e: rusqlite::Error| PyRuntimeError::new_err(format!("Update failed: {}", e)))?;

    conn.execute(
        "DELETE FROM weight_storage
         WHERE memory_name = ?1 AND model_type = 'lstm' AND is_active = 0",
        params![memory_name],
    ).map_err(|e: rusqlite::Error| PyRuntimeError::new_err(format!("Cleanup failed: {}", e)))?;

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
fn load_lstm_weights(db_path: String, memory_name: String) -> PyResult<Option<String>> {
    let conn: Connection = Connection::open(&db_path)
        .map_err(|e: rusqlite::Error| PyRuntimeError::new_err(format!("DB open failed: {}", e)))?;

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
        Err(e) => Err(PyRuntimeError::new_err(format!("Query failed: {}", e))),
    }
}

#[pyfunction]
fn save_transformer_weights(
    db_path: String,
    memory_name: String,
    binary_data: Vec<u8>,
) -> PyResult<()> {
    let conn: Connection = Connection::open(&db_path)
        .map_err(|e: rusqlite::Error| PyRuntimeError::new_err(format!("DB open failed: {}", e)))?;

    conn.execute(
        "INSERT INTO weight_storage (memory_name, model_type, weights, is_active)
         VALUES (?1, 'transformer', ?2, 1)",
        params![memory_name, binary_data],
    ).map_err(|e: rusqlite::Error| PyRuntimeError::new_err(format!("Insert failed: {}", e)))?;

    conn.execute(
        "UPDATE weight_storage SET is_active = 0
         WHERE memory_name = ?1 AND model_type = 'transformer'
         AND id != last_insert_rowid()",
        params![memory_name],
    ).map_err(|e: rusqlite::Error| PyRuntimeError::new_err(format!("Update failed: {}", e)))?;

    conn.execute(
        "DELETE FROM weight_storage
         WHERE memory_name = ?1 AND model_type = 'transformer' AND is_active = 0",
        params![memory_name],
    ).map_err(|e: rusqlite::Error| PyRuntimeError::new_err(format!("Cleanup failed: {}", e)))?;

    Ok(())
}

#[pyfunction]
fn load_transformer_weights(db_path: String, memory_name: String) -> PyResult<Option<Vec<u8>>> {
    let conn: Connection = Connection::open(&db_path)
        .map_err(|e: rusqlite::Error| PyRuntimeError::new_err(format!("DB open failed: {}", e)))?;

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
        Err(e) => Err(PyRuntimeError::new_err(format!("Query failed: {}", e))),
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
    Ok(())
}
