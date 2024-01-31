
use super::cu;

const PTX_DIR: &str = env!("OUT_DIR");

pub fn from_ptx(name: &str) -> cu::Module {

    let path = format!("{}{}.ptx", PTX_DIR, name);

    todo!()
}