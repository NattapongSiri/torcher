use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Generate extern C function that map directly to Caffe2 function
/// # Parameters
/// - dt - A slice of tuple with pair of datatype and struct name
/// - f - A mutable file to be written
fn generate_caffe2_fn(dt: &[(&str, &str)], f: &mut File) {
    f.write_all(b"
        #[link(name=\"caffe2\")]
        extern {
    ").expect("Fail to write link to Caffe2 header");

    for pref in dt {
        f.write_all(format!("
            fn {pref}_new() -> *mut TorchStorage;
            fn {pref}_newWithSize(size: i64) -> *mut TorchStorage;
            fn {pref}_data(storage: *mut TorchStorage) -> *mut {t};
            fn {pref}_fill(storage: *mut TorchStorage, scalar: {t});
            fn {pref}_resize(storage: *mut TorchStorage, size: i64);
            fn {pref}_retain(storage: *mut TorchStorage);
            fn {pref}_size(storage: *const TorchStorage) -> i64;
            fn {pref}_swap(storage_1: *mut TorchStorage, storage_2: *mut TorchStorage);
            fn {pref}_free(storage: *mut TorchStorage);
            ",
            pref=pref.1,
            t=pref.0
        ).as_bytes()).expect(format!("Fail to generate Caffe2 function for {}", pref.1).as_str());
    }
    f.write_all(b"}").expect("Fail to close the brace");
}

fn impl_storages(dt: &[(&str, &str)], f: &mut File) {
    for ty in dt {
        f.write_all(format!("
            impl StorageOp<{t}> for Storage<{t}> {{
                fn new() -> Storage<{t}> {{
                    unsafe {{
                        let sys_storage = {pref}_new();
                        let sys_data = {pref}_data(sys_storage);
                        let size = {pref}_size(sys_storage);
                        
                        Storage {{
                            data: std::slice::from_raw_parts_mut(sys_data, size as usize) as *mut [{t}],
                            forget: false,
                            size: 0,
                            storage: sys_storage
                        }}
                    }}
                }}
                fn new_with_size(size: usize) -> Storage<{t}> {{
                    unsafe {{
                        let sys_storage = {pref}_newWithSize(size as i64);
                        let sys_data = {pref}_data(sys_storage);
                        let actual_size = {pref}_size(sys_storage);
                        
                        Storage {{
                            data: std::slice::from_raw_parts_mut(sys_data, actual_size as usize) as *mut [{t}],
                            forget: false,
                            size: size,
                            storage: sys_storage
                        }}
                    }}
                }}
                fn data(&self) -> &[{t}] {{
                    unsafe {{
                        &*self.data
                    }}
                }}
                fn data_mut(&mut self) -> &mut [{t}] {{
                    unsafe {{
                        &mut *self.data
                    }}
                }}
                fn forget(&mut self) {{
                    self.forget = true;
                }}
                fn fill(&mut self, scalar: {t}) {{
                    self.data_mut().iter_mut().for_each(|v| *v = scalar);
                }}
                fn resize(&mut self, size: usize) {{
                    unsafe {{
                        {pref}_resize(self.storage, size as i64);
                        self.size = size;
                    }}
                }}
                fn retain(&mut self) {{
                    unsafe {{
                        {pref}_retain(self.storage);
                    }}
                }}
                fn size(&self) -> usize {{
                    self.size
                }}
                fn swap(&mut self, other: &mut Storage<{t}>) {{
                    {pref}_swap(self.storage, other.storage);
                }}
                fn free(&mut self) {{
                    {pref}_free(self.storage);
                }}
            }}
        ",
        pref=ty.1,
        t=ty.0).as_bytes()).expect(format!("Fail write \"impl Storage<{}>\"", ty.0).as_str());
    }
}

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("sys.rs");
    let mut f = File::create(&dest_path).unwrap();
    let caffe2_prefix = [
        ("u8", "THByteStorage"), 
        ("i8", "THCharStorage"),
        ("f32", "THFloatStorage"),
        ("f64", "THDoubleStorage"),
        ("i32", "THIntStorage"),
        ("i64", "THLongStorage"),
        ("i16", "THShortStorage")
    ];
    generate_caffe2_fn(&caffe2_prefix, &mut f);
    impl_storages(&caffe2_prefix, &mut f);
    println!(r"cargo:rustc-link-search=clib");
}