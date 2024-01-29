

use std::sync::Mutex;
use std::sync::MutexGuard;

/// Lazily Initialized Static Value
pub struct Lazy<T> {
    value: Mutex<LazyValue<T>>,
    init: fn() -> T,
}

impl<T> Lazy<T> {
    pub const fn new(init: fn() -> T) -> Self {
        Self {
            value: Mutex::new(LazyValue::none()),
            init
        }
    }

    pub fn get(&self) -> MutexGuard<'_, LazyValue<T>> {
        let mut lock = self.value.lock().unwrap();

        if lock.is_none() {
            lock.set((self.init)())
        }

        lock
    }
}

pub struct LazyValue<T> {
    v: Option<T>
}

impl<T> LazyValue<T> {
    const fn none() -> Self {
        Self {
            v: None
        }
    }

    fn is_none(&self) -> bool {
        self.v.is_none()
    }

    fn set(&mut self, value: T) {
        self.v = Some(value)
    }
}

impl<T> std::ops::Deref for LazyValue<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.v.as_ref().unwrap()
    }
}

impl<T> std::ops::DerefMut for LazyValue<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.v.as_mut().unwrap()
    }
}