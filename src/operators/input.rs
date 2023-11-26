
use super::*;

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct Input;

impl Operator for Input {
    fn forward(&mut self, _: &Node) -> Result<()> {
        Err(anyhow!("Input Operator forwards should never be executed"))
    }

    fn backward(&mut self, _: &Node) -> Result<()> {
        Err(anyhow!("Input Operator backwards should never be executed"))
    }
}

impl Display for Input {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Input")
    }
}