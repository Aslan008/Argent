use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::mpsc::Sender;
use std::time::Instant;
use crate::events::{parse_py_event, ArgentEvent};

/// An event and the moment it reached the sink. That is the moment it happened as
/// far as the UI can tell; when the UI gets to draw it is later and uneven, which is
/// no basis for timing a stream.
pub type Stamped = (Instant, ArgentEvent);

#[pyclass]
pub struct RustEventSink {
    pub tx: Sender<Stamped>,
}

#[pymethods]
impl RustEventSink {
    #[pyo3(signature = (event))]
    fn __call__(&self, event: &Bound<'_, PyDict>) -> PyResult<()> {
        let at = Instant::now();
        if let Ok(Some(ev)) = parse_py_event(event) {
            let _ = self.tx.send((at, ev));
        }
        Ok(())
    }
}
