
pub struct Unsafe<T>(T);

impl<T> Unsafe<T> {
    pub fn new(v: T) -> Self {
        Self(v)
    }

    pub fn get(&self) -> &T {
        unsafe {
            #[allow(invalid_reference_casting)]
            & *(&self.0 as *const T)
        }
    }

    pub fn get_mut(&self) -> &mut T {
        unsafe {
            #[allow(invalid_reference_casting)]
            &mut *(&self.0 as *const T as *mut T) 
        }
    }
}

impl<T> std::ops::Deref for Unsafe<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.get()
    }
}

impl<T> std::ops::DerefMut for Unsafe<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.get_mut()
    }
}