#![recursion_limit="1024"]

extern crate common;
extern crate proc_macro;
extern crate proc_macro2;
extern crate syn;
extern crate quote;

use common::{StorageType};
use proc_macro::{TokenStream};
use proc_macro2::{Span};
use syn::{Data, DeriveInput, Ident, Meta, parse_macro_input};
use quote::quote;

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
    let size_fn = Ident::new(&(c_ty.to_owned() + "_size"), Span::call_site());
    let swap_fn = Ident::new(&(c_ty.to_owned() + "_swap"), Span::call_site());
    let free_fn = Ident::new(&(c_ty.to_owned() + "_free"), Span::call_site());

    let expanded = quote! {
        #[repr(C)]
        pub struct #c_ty_id;
        
        #[link(name="caffe2")]
        extern {
            fn #new_fn() -> *mut #c_ty_id;
            fn #new_with_size_fn(size: usize) -> *mut #c_ty_id;
            fn #data_fn(storage: *mut #c_ty_id) -> *mut #t;
            fn #fill_fn(storage: *mut #c_ty_id, scalar: #t);
            fn #resize_fn(storage: *mut #c_ty_id, size: usize);
            fn #retain_fn(storage: *mut #c_ty_id);
            fn #size_fn(storage: *const #c_ty_id) -> usize;
            fn #swap_fn(storage_1: *mut #c_ty_id, storage_2: *mut #c_ty_id);
            fn #free_fn(storage: *mut #c_ty_id);
        }

        pub struct #ident<'a> {
            _data: &'a mut [#t],
            forget : bool,
            storage : *mut #c_ty_id,
            n : usize
        }

        impl<'a> #ident<'a> {
            /// Get short description of storage.
            /// This includes name of storage, size, and
            /// sample data if it has more than 20 elements.
            /// If it has less than 20 elements, it'll display
            /// every elements.
            fn short_desc(&mut self) -> String {
                let size = self.n;
                let data = &self._data;
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

            /// Unsafe function to retrieve underlying pointer to actual storage.
            pub unsafe fn storage(&self) -> *mut #c_ty_id {
                self.storage
            }

            /// Alias for short_desc
            #[inline(always)]
            fn to_string(&mut self) -> String {
                self.short_desc()
            }
        }

        impl<'a> TensorStorage for #ident<'a> {
            type Datum = #t;
            
            fn new() -> Self {
                unsafe {
                    let storage = #new_fn();
                    let size = #size_fn(storage);

                    #ident {
                        _data: std::slice::from_raw_parts_mut(#data_fn(storage), size),
                        forget: false,
                        storage: storage,
                        n: size
                    }
                }
            }

            fn new_with_size(size: usize) -> Self {
                unsafe {
                    let storage = #new_with_size_fn(size);
                    
                    #ident {
                        _data: std::slice::from_raw_parts_mut(#data_fn(storage), size),
                        forget: false,
                        storage: storage,
                        n: size
                    }
                }
            }

            fn data(&self) -> &[#t] {
                self._data
            }

            fn data_mut(&mut self) -> &mut [#t] {
                self._data
            }

            unsafe fn forget(mut self) -> Self {
                self.forget = true;
                self
            }

            fn fill(&mut self, value: #t) {
                self._data.iter_mut().for_each(|v| *v = value);
            }

            fn resize(&mut self, size: usize) {
                self.n = size;

                unsafe {
                    #resize_fn(self.storage, size);
                    self._data = std::slice::from_raw_parts_mut(#data_fn(self.storage), size);
                }
            }

            fn retain(&mut self) {
                unsafe {
                    #retain_fn(self.storage);
                    self.n = #size_fn(self.storage);
                    self._data = std::slice::from_raw_parts_mut(#data_fn(self.storage), self.n);
                }
            }

            fn size(&self) -> usize {
                self.n
            }

            fn swap(&mut self, with : Self) {
                unsafe {
                    // need to call to Caffe2 to ensure that their storage got swap as well
                    #swap_fn(self.storage, with.storage);
                    self.n = with.n;
                    self._data = std::slice::from_raw_parts_mut(#data_fn(self.storage), self.n);
                }
            }
        }

        /// For each of usage, it return a slice of actual data
        /// of this struct. For higher throughput, consider using 
        /// [data function](trait.TensorStorage.html#tymethod.data) instead.
        impl<'a> Deref for #ident<'a> {
            type Target = [#t];

            fn deref(&self) -> &[#t] {
                self._data
            }
        }

        /// For each of usage, it return mutable slice of actual data
        /// of this struct. For higher throughput, consider using 
        /// [data function](trait.TensorStorage.html#tymethod.data) instead.
        impl<'a> DerefMut for #ident<'a> {
            fn deref_mut(&mut self) -> &mut [#t] {
                self._data
            }
        }

        /// Clean up memory allocated outside of Rust.
        /// Unless [forget function](trait.TensorStorage.html#tymethod.forget) is called,
        /// it'll leave underlying storage untouch.
        impl<'a> Drop for #ident<'a> {
            fn drop(&mut self) {
                unsafe {
                    if !self.forget {
                        #free_fn(self.storage);
                    }
                }
            }
        }

        impl<'a> From<*mut #c_ty_id> for #ident<'a> {
            fn from(c_storage: *mut #c_ty_id) -> #ident<'a> {
                unsafe {
                    let size = #size_fn(c_storage);
                    let data = #data_fn(c_storage);

                    #ident {
                        _data: std::slice::from_raw_parts_mut(data, size),
                        forget: false, 
                        storage: c_storage,
                        n: size
                    }
                }
            }
        }
    };

    TokenStream::from(expanded)
}
