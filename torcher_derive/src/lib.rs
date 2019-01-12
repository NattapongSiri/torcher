#![recursion_limit="1024"]

extern crate proc_macro;
extern crate proc_macro2;
extern crate syn;
extern crate quote;

use proc_macro::{TokenStream};
use proc_macro2::{Span};
use std::fmt;
use std::fmt::{Display};
use syn::{Data, DeriveInput, Ident, Meta, parse_macro_input};
use quote::quote;

/// An enum representing Storage type
enum StorageType {
    // float group
    THFloatStorage(&'static str),
    THDoubleStorage(&'static str),

    // byte group
    THByteStorage(&'static str),
    THCharStorage(&'static str),

    // integer group
    THShortStorage(&'static str),
    THIntStorage(&'static str),
    THLongStorage(&'static str)
}

impl StorageType {
    /// Construct the enum from string.
    pub fn from_str(def : &str) -> StorageType {
        match def {
            "THFloatStorage" | "f32" | "float" => {
                StorageType::THFloatStorage("f32")
            },
            "THDoubleStorage" | "f64" | "double" => {
                StorageType::THDoubleStorage("f64")
            },
            "THByteStorage" | "u8" | "byte" => {
                StorageType::THByteStorage("u8")
            },
            "THCharStorage" | "i8" | "char" => {
                StorageType::THCharStorage("i8")
            },
            "THShortStorage" | "i16" | "short" => {
                StorageType::THShortStorage("i16")
            },
            "THIntStorage" | "i32" | "int" => {
                StorageType::THIntStorage("i32")
            },
            "THLongStorage" | "i64" | "long" => {
                StorageType::THLongStorage("i64")
            },
            _ => {
                panic!("Unsupported storage type.")
            }
        }
    }

    /// Get a rust primitive type for current storage type.
    pub fn to_ty(&self) -> &'static str {
        match self {
            StorageType::THFloatStorage(ty) |
            StorageType::THDoubleStorage(ty) |
            StorageType::THByteStorage(ty) |
            StorageType::THCharStorage(ty) | 
            StorageType::THShortStorage(ty) |
            StorageType::THIntStorage(ty) |
            StorageType::THLongStorage(ty) => {
                ty
            },
        }
    }
}

impl Display for StorageType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            StorageType::THFloatStorage(_) => {
                write!(f, "THFloatStorage")
            },
            StorageType::THDoubleStorage(_) => {
                write!(f, "THDoubleStorage")
            },
            StorageType::THByteStorage(_) => {
                write!(f, "THByteStorage")
            },
            StorageType::THCharStorage(_) => {
                write!(f, "THCharStorage")
            },
            StorageType::THShortStorage(_) => {
                write!(f, "THShortStorage")
            },
            StorageType::THIntStorage(_) => {
                write!(f, "THIntStorage")
            },
            StorageType::THLongStorage(_) => {
                write!(f, "THLongStorage")
            },
        }
    }
}

#[allow(non_snake_case)]
#[proc_macro_attribute]
pub fn TorchStorage(args : TokenStream, item : TokenStream) -> TokenStream {
    if args.is_empty() {
        panic!("
                Need exactly one of variant name defined in StorageType enum.
                For example:
                #[TensorStorage(THFloatStorage)]
                struct FloatStorage {}
            ")
    }
    let ty = match parse_macro_input!(args) {
        
        Meta::Word(ident) => {
            StorageType::from_str(&ident.to_string())
        },
        _ => {
            panic!("
                Need exactly one of variant name defined in StorageType enum.
                For example:
                #[TensorStorage(THFloatStorage)]
                struct FloatStorage {}
            ")
        }
        // Meta::List(idents) => {
        //     print!("Got list=[");
        //     idents.nested.iter().for_each(|meta| {
        //         if let NestedMeta::Meta(Meta::Word(id)) = meta {
        //             print!("{}, ", id);
        //         } else if let NestedMeta::Literal(Lit::Str(id)) = meta {
        //             print!("\"{}\", ", id.value());
        //         } else {
        //             print!("UNDEFINED");
        //         }
        //     });
        //     println!("]")
        // },
        // Meta::NameValue(pairs) => {
        //     if let Lit::Str(v) = pairs.lit {
        //         println!("Got {}={}", pairs.ident, v.value());
        //     } else {
        //         println!("Got unsupported pair");
        //     }
        // }
    };

    let c_ty = ty.to_string();
    let c_ty_id = Ident::new(&c_ty, Span::call_site());

    let ast = parse_macro_input!(item as DeriveInput);
    match ast.data {
        Data::Struct(_) => (),
        _ => panic!("
            This procedural macro support only struct.
            For example:
            #[TorchStorage(i8)]
            struct ByteStorage;
        ")
    }
    let ident = ast.ident;
    let t = Ident::new(ty.to_ty(), Span::call_site());

    let new_fn = Ident::new(&(c_ty.to_owned() + "_new"), Span::call_site());
    let new_with_size_fn = Ident::new(&(c_ty.to_owned() + "_newWithSize"), Span::call_site());
    let data_fn = Ident::new(&(c_ty.to_owned() + "_data"), Span::call_site());
    let fill_fn = Ident::new(&(c_ty.to_owned() + "_fill"), Span::call_site());
    let resize_fn = Ident::new(&(c_ty.to_owned() + "_resize"), Span::call_site());
    let retain_fn = Ident::new(&(c_ty.to_owned() + "_retain"), Span::call_site());
    let swap_fn = Ident::new(&(c_ty.to_owned() + "_swap"), Span::call_site());
    let free_fn = Ident::new(&(c_ty.to_owned() + "_free"), Span::call_site());

    let expanded = quote! {
        #[repr(C)]
        struct #c_ty_id;
        
        #[link(name="caffe2")]
        extern {
            fn #new_fn() -> *mut #c_ty_id;
            fn #new_with_size_fn(size: usize) -> *mut #c_ty_id;
            fn #data_fn(storage: *mut #c_ty_id) -> *mut #t;
            fn #fill_fn(storage: *mut #c_ty_id, scalar: #t);
            fn #resize_fn(storage: *mut #c_ty_id, size: usize);
            fn #retain_fn(storage: *mut #c_ty_id);
            fn #swap_fn(storage_1: *mut #c_ty_id, storage_2: *mut #c_ty_id);
            fn #free_fn(storage: *mut #c_ty_id);
        }

        struct #ident {
            forget : bool,
            storage : *mut #c_ty_id,
            n : usize
        }

        impl #ident {
            /// Get short description of storage.
            /// This includes name of storage, size, and
            /// sample data if it has more than 20 elements.
            /// If it has less than 20 elements, it'll display
            /// every elements.
            fn short_desc(&mut self) -> String {
                let size = self.n;
                let data = &self.data();
                let name = stringify!(#ident);

                if size > 20 {
                    format!("{}:size={}:first(10)={:?}:last(10)={:?}", name, size,
                        &data[0..10], &data[(data.len() - 10)..data.len()]
                    )
                } else {
                    format!("{}:size={}:data={:?}", name, size,
                        data
                    )
                }
            }

            /// Alias for short_desc
            #[inline(always)]
            fn to_string(&mut self) -> String {
                self.short_desc()
            }
        }

        impl TensorStorage<#t> for #ident {
            fn new() -> Self {
                unsafe {
                    #ident {
                        forget: false,
                        storage: #new_fn(),
                        n: 0
                    }
                }
            }

            fn new_with_size(size: usize) -> Self {
                unsafe {
                    #ident {
                        forget: false,
                        storage: #new_with_size_fn(size),
                        n: size
                    }
                }
            }

            fn data<'a>(&'a mut self) -> &'a mut [#t] {
                unsafe {
                    std::slice::from_raw_parts_mut::<'a>(
                        #data_fn(self.storage), 
                        self.n
                    )
                }
            }

            unsafe fn forget(mut self) {
                self.forget = true;
            }

            fn fill(&mut self, value: #t) {
                unsafe {
                    #fill_fn(self.storage, value);
                }
            }

            fn resize(&mut self, size: usize) {
                self.n = size;

                unsafe {
                    #resize_fn(self.storage, size);
                }
            }

            fn retain(&mut self) {
                unsafe {
                    #retain_fn(self.storage);
                }
            }

            fn size(&self) -> usize {
                self.n
            }

            fn swap(&mut self, with : Self) {
                unsafe {
                    #swap_fn(self.storage, with.storage);
                }
            }
        }

        impl Drop for #ident {
            fn drop(&mut self) {
                unsafe {
                    if !self.forget {
                        #free_fn(self.storage);
                    }
                }
            }
        }
    };

    TokenStream::from(expanded)
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
