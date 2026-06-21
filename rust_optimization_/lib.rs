// src/lib.rs
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use rusqlite::{Connection, params};

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
    Ok(())
}