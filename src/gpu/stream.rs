
use super::cu;

#[derive(Copy, Clone)]
pub struct Stream {
    pub(crate) stream: cu::Stream,
}

impl Stream {
    pub fn null() -> Self {
        Self {
            stream: cu::Stream::null()
        }
    }
}