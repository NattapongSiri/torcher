use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;

fn implement_type_conversion_for_trait() {
    fn gen_code(src: &str, src_data: &str, dest: &str, dest_store: &str, dest_ty: &str, storage_expr: &str) -> String {
        format!("
            impl<'a> From<&'a {src}> for {dest}
            {{
                /// Perform type casting from {src} to {dest}.
                /// This is done by deep cloning on entire storage while also
                /// casting each element in storage from `u8` to `i8` at the same time.
                /// The return tensor is completely independent from original tensor.
                fn from(src: &'a {src}) -> {dest} {{
                    let (size, stride) = src.shape();
                    let src_data = {src_data};
                    let mut storage = {dest_storage}::new_with_size(src_data.len());
                    let data = storage.data_mut();

                    data.iter_mut().zip(src_data.iter()).for_each(|(dest, src)| *dest = *src as {dest_ty});
                    return {dest}::new_with_storage_nd({storage_expr}, 0, size, stride);
                }}
            }}
            ",
            src=src,
            src_data=src_data,
            storage_expr=storage_expr,
            dest=dest,
            dest_storage=dest_store,
            dest_ty=dest_ty 
        )
    }
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("conv.rs");
    let mut f = File::create(&dest_path).unwrap();
    let supported_types = [
        ("ByteTensor", "ByteStorage", "u8"), 
        ("CharTensor", "CharStorage", "i8"),
        ("FloatTensor", "FloatStorage", "f32"),
        ("DoubleTensor", "DoubleStorage", "f64"),
        ("IntTensor", "IntStorage", "i32"),
        ("LongTensor", "LongStorage", "i64"),
        ("ShortTensor", "ShortStorage", "i16")
    ];

    for src in &supported_types {
        for dest in &supported_types {
            if src != dest {
                if cfg!(feature = "safe") {
                    f.write_all(
                        gen_code(
                            src.0,
                            "src.storage.as_ref().unwrap().borrow()",
                            dest.0,
                            dest.1,
                            dest.2,
                            "Rc::new(RefCell::new(storage))"
                        ).as_str()
                     .as_bytes())
                     .expect("Auto implementation error on implementing From for each tensor");
                } else if cfg!(feature = "threadsafe") {
                    f.write_all(
                        gen_code(
                            src.0,
                            "src.storage.as_ref().unwrap().read().unwrap()",
                            dest.0,
                            dest.1,
                            dest.2,
                            "Arc::new(RwLock::new(storage))"
                        ).as_str()
                     .as_bytes())
                     .expect("Auto implementation error on implementing From for each tensor");
                } else if cfg!(feature = "unsafe") {
                    f.write_all(
                        gen_code(
                            src.0,
                            "src.data()",
                            dest.0,
                            dest.1,
                            dest.2,
                            "storage"
                        ).as_str()
                     .as_bytes())
                     .expect("Auto implementation error on implementing From for each tensor");
                }
            }
        }
    }
}

fn main() {
    // implement auto conversion
    implement_type_conversion_for_trait();

    println!(r"cargo:rustc-link-search=clib");
}