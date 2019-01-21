#![recursion_limit="1024"]

extern crate common;
extern crate proc_macro;
extern crate proc_macro2;
extern crate syn;
extern crate quote;

use common::{StorageType, TensorType};
use proc_macro::{TokenStream};
use proc_macro2::{Span};
use syn::{Data, DeriveInput, Ident, Lit, Meta, parse_macro_input};
use quote::quote;

fn _make_ident(pref: &str, name: &str) -> Ident {
    Ident::new(&(pref.to_owned() + name), Span::call_site())
}

#[allow(non_snake_case)]
#[proc_macro_attribute]
pub fn TorchTensor(args : TokenStream, item : TokenStream) -> TokenStream {
    if args.is_empty() {
        panic!("
                Need exactly one of variant name defined in TensorType enum.
                For example:
                #[TorchTensor(THFloatTensor)]
                struct FloatTensor;
            ")
    }

    let (ty, store_ty) = match parse_macro_input!(args) {
        Meta::NameValue(pairs) => {
            if let Lit::Str(v) = pairs.lit {
                (pairs.ident, v.value())
            } else {
                panic!("
                Need a name of struct that implement TorchStorage.
                For example:
                #[TorchTensor(f32=FloatStorage)]
                struct FloatTensor;
                ")
            }
        }
        _ => {
            panic!("
                Need exactly one of variant name defined in TensorType enum.
                For example:
                #[TorchTensor(f32=FloatStorage)]
                struct FloatTensor;
            ")
        }
    };

    let c_ty = TensorType::from_str(&ty.to_string()).to_string();
    let c_ty_id = Ident::new(&c_ty, Span::call_site());

    let store_ty_id = Ident::new(&store_ty, Span::call_site());
    let c_storage_ty = StorageType::from_str(&ty.to_string()).to_string();
    let c_storage_ty_id = Ident::new(&c_storage_ty, Span::call_site());

    let ast = parse_macro_input!(item as DeriveInput);
    match ast.data {
        Data::Struct(_) => (),
        _ => panic!("
            This procedural macro support only struct.
            For example:
            #[TorchTensor(i8)]
            struct ByteStorage;
        ")
    }

    let ident = ast.ident;
    let t = ty;

    let new_fn = _make_ident(&c_ty, "_new");
    let new_clone_fn = _make_ident(&c_ty, "_newClone");
    let new_contiguous_fn = _make_ident(&c_ty, "_newContiguous");
    let new_select_fn = _make_ident(&c_ty, "_newSelect");
    let new_narrow_fn = _make_ident(&c_ty, "_newNarrow");
    let new_transpose_fn = _make_ident(&c_ty, "_newTranspose");
    let new_unfold_fn = _make_ident(&c_ty, "_newUnfold");
    let new_with_tensor = _make_ident(&c_ty, "_newWithTensor");
    let new_with_storage_1d_fn = _make_ident(&c_ty, "_newWithStorage1d");
    let new_with_storage_2d_fn = _make_ident(&c_ty, "_newWithStorage2d");
    let new_with_storage_3d_fn = _make_ident(&c_ty, "_newWithStorage3d");
    let new_with_storage_4d_fn = _make_ident(&c_ty, "_newWithStorage4d");
    let new_with_size_1d_fn = _make_ident(&c_ty, "_newWithSize1d");
    let new_with_size_2d_fn = _make_ident(&c_ty, "_newWithSize2d");
    let new_with_size_3d_fn = _make_ident(&c_ty, "_newWithSize3d");
    let new_with_size_4d_fn = _make_ident(&c_ty, "_newWithSize4d");
    let free_fn = _make_ident(&c_ty, "_free");

    let data_fn = _make_ident(&c_ty, "_data");
    let desc_fn = _make_ident(&c_ty, "_desc");
    let dim_fn = _make_ident(&c_ty, "_nDimension");
    let is_contiguous_fn = _make_ident(&c_ty, "_isContiguous");
    let get_0d_fn = _make_ident(&c_ty, "_get0d");
    let get_1d_fn = _make_ident(&c_ty, "_get1d");
    let get_2d_fn = _make_ident(&c_ty, "_get2d");
    let get_3d_fn = _make_ident(&c_ty, "_get3d");
    let get_4d_fn = _make_ident(&c_ty, "_get4d");
    let numel_fn = _make_ident(&c_ty, "_numel");
    let resize_0d_fn = _make_ident(&c_ty, "_resize0d");
    let resize_1d_fn = _make_ident(&c_ty, "_resize1d");
    let resize_2d_fn = _make_ident(&c_ty, "_resize2d");
    let resize_3d_fn = _make_ident(&c_ty, "_resize3d");
    let resize_4d_fn = _make_ident(&c_ty, "_resize4d");
    let resize_5d_fn = _make_ident(&c_ty, "_resize5d");
    let resize_nd_fn = _make_ident(&c_ty, "_resizeNd");
    let set_0d_fn = _make_ident(&c_ty, "_set0d");
    let set_1d_fn = _make_ident(&c_ty, "_set1d");
    let set_2d_fn = _make_ident(&c_ty, "_set2d");
    let set_3d_fn = _make_ident(&c_ty, "_set3d");
    let set_4d_fn = _make_ident(&c_ty, "_set4d");
    let size_fn = _make_ident(&c_ty, "_size");
    let storage_fn = _make_ident(&c_ty, "_storage");
    let storage_offset_fn = _make_ident(&c_ty, "_storageOffset");
    let stride_fn = _make_ident(&c_ty, "_stride");
    let squeeze_fn = _make_ident(&c_ty, "_squeeze");

    let expanded = quote! {
        use storage::#c_storage_ty_id;

        #[repr(C)]
        pub struct #c_ty_id;
        
        #[link(name="caffe2")]
        extern {
            fn #new_fn() -> *mut #c_ty_id;
            fn #new_clone_fn(org: *const #c_ty_id) -> *mut #c_ty_id;
            fn #new_contiguous_fn(org: *const #c_ty_id) -> *mut #c_ty_id;
            fn #new_narrow_fn(org: *const #c_ty_id, dim: c_int, i: i64, size: i64) -> *mut #c_ty_id;
            fn #new_select_fn(org: *const #c_ty_id, dim: c_int, idx: i64) -> *mut #c_ty_id;
            fn #new_transpose_fn(org: *const #c_ty_id, dim_1: c_int, dim_2: c_int) -> *mut #c_ty_id;
            fn #new_unfold_fn(org: *const #c_ty_id, dim: c_int, size: i64, step: i64) -> *mut #c_ty_id;
            fn #new_with_tensor(org: *const #c_ty_id) -> *mut #c_ty_id;
            fn #new_with_storage_1d_fn(store : *mut #c_storage_ty_id, offset: usize, size: usize, stride: usize) -> *mut #c_ty_id;
            fn #new_with_storage_2d_fn(store : *mut #c_storage_ty_id, offset: usize, size_1: usize, stride_1: usize
                                                                               , size_2: usize, stride_2: usize) -> *mut #c_ty_id;
            fn #new_with_storage_3d_fn(store : *mut #c_storage_ty_id, offset: usize, size_1: usize, stride_1: usize
                                                                               , size_2: usize, stride_2: usize
                                                                               , size_3: usize, stride_3: usize) -> *mut #c_ty_id;
            fn #new_with_storage_4d_fn(store : *mut #c_storage_ty_id, offset: usize, size_1: usize, stride_1: usize
                                                                               , size_2: usize, stride_2: usize
                                                                               , size_3: usize, stride_3: usize
                                                                               , size_4: usize, stride_4: usize) -> *mut #c_ty_id;
            fn #new_with_size_1d_fn(size: i64) -> *mut #c_ty_id;
            fn #new_with_size_2d_fn(size_1: i64, size_2: i64) -> *mut #c_ty_id;
            fn #new_with_size_3d_fn(size_1: i64, size_2: i64, size_3: i64) -> *mut #c_ty_id;
            fn #new_with_size_4d_fn(size_1: i64, size_2: i64, size_3: i64, size_4: i64) -> *mut #c_ty_id;
            fn #free_fn(tensor: *mut #c_ty_id);

            fn #data_fn(tensor: *mut #c_ty_id) -> *mut #t;
            fn #desc_fn(tensor: *mut #c_ty_id) -> THDescBuff;
            fn #dim_fn(tensor: *const #c_ty_id) -> c_int;
            fn #is_contiguous_fn(tensor: *const #c_ty_id) -> c_int;
            fn #get_0d_fn(tensor: *const #c_ty_id) -> #t;
            fn #get_1d_fn(tensor: *const #c_ty_id, i: i64) -> #t;
            fn #get_2d_fn(tensor: *const #c_ty_id, i: i64, j: i64) -> #t;
            fn #get_3d_fn(tensor: *const #c_ty_id, i: i64, j: i64, k: i64) -> #t;
            fn #get_4d_fn(tensor: *const #c_ty_id, i: i64, j: i64, k: i64, l: i64) -> #t;
            fn #numel_fn(tensor: *const #c_ty_id) -> usize;
            fn #resize_0d_fn(tensor: *mut #c_ty_id);
            fn #resize_1d_fn(tensor: *mut #c_ty_id, size_1: i64);
            fn #resize_2d_fn(tensor: *mut #c_ty_id, size_1: i64, size_2: i64);
            fn #resize_3d_fn(tensor: *mut #c_ty_id, size_1: i64, size_2: i64, size_3: i64);
            fn #resize_4d_fn(tensor: *mut #c_ty_id, size_1: i64, size_2: i64, size_3: i64, size_4: i64);
            fn #resize_5d_fn(tensor: *mut #c_ty_id, size_1: i64, size_2: i64, size_3: i64, size_4: i64, size_5: i64);
            fn #resize_nd_fn(tensor: *mut #c_ty_id, dim: c_int, size: *const i64, stride: *const i64);
            fn #set_0d_fn(tensor: *const #c_ty_id, v: #t);
            fn #set_1d_fn(tensor: *const #c_ty_id, i: i64, v: #t);
            fn #set_2d_fn(tensor: *const #c_ty_id, i: i64, j: i64, v: #t);
            fn #set_3d_fn(tensor: *const #c_ty_id, i: i64, j: i64, k: i64, v: #t);
            fn #set_4d_fn(tensor: *const #c_ty_id, i: i64, j: i64, k: i64, l: i64, v: #t);
            fn #size_fn(tensor: *const #c_ty_id, dim: c_int) -> i64;
            fn #storage_fn(tensor: *mut #c_ty_id) -> *mut #c_storage_ty_id;
            fn #storage_offset_fn(tensor: *const #c_ty_id) -> i64;
            fn #stride_fn(tensor: *const #c_ty_id, dim: c_int) -> i64;
            fn #squeeze_fn(tensor: *mut #c_ty_id, res: *mut #c_ty_id);
        }

        pub struct #ident {
            forget : bool,
            storage : Option<#store_ty_id>,
            tensor : *mut #c_ty_id,
            size : Vec<usize>,
            stride : Vec<usize>
        }

        impl #ident {
            /// Get short description of storage.
            /// This includes name of storage, size, and
            /// sample data if it has more than 20 elements.
            /// If it has less than 20 elements, it'll display
            /// every elements.
            fn short_desc(&mut self) -> String {
                let size = self.size.as_slice();
                let stride = self.stride.as_slice();
                let data = match self.storage {
                    Some(ref mut store) => &*store.data(),
                    None => &[]
                };
                let name = stringify!(#ident);

                if size.iter().fold(0, |cum, v| cum + v) > 20 {
                    format!("{}:size={:?}:stride={:?}:first(10)={:?}:last(10)={:?}", name, size,
                        stride, &data[0..10], &data[(data.len() - 10)..data.len()]
                    )
                } else {
                    format!("{}:size={:?}:stride={:?}:data={:?}", name, size, stride,
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

        impl Tensor<#t> for #ident {
            type Storage = #store_ty_id;

            fn new() -> Self {
                unsafe {
                    #ident {
                        forget: false,
                        storage: None,
                        tensor: #new_fn(),
                        size: Vec::new(),
                        stride: Vec::new()
                    }
                }
            }

            fn new_contiguous(&self) -> Self {
                unsafe {
                    let cont = #new_contiguous_fn(self.tensor);
                    let storage: #store_ty_id = #storage_fn(cont).into();
                    let stride : Vec<usize> = (0..#dim_fn(cont)).map(|i| {
                        #stride_fn(cont, i as i32) as usize
                    }).collect();

                    #ident {
                        forget: false,
                        storage: Some(storage.forget()),
                        tensor: cont,
                        size: self.size.to_owned(),
                        stride: stride
                    }
                }
            }

            unsafe fn new_narrow(&self, dim: usize, i: usize, size: usize) -> Self {
                unsafe {
                    let tensor = #new_narrow_fn(self.tensor, dim as c_int, i as i64, size as i64);
                    let storage: #store_ty_id = #storage_fn(tensor).into();
                    let stride : Vec<usize> = (0..#dim_fn(tensor)).map(|i| {
                        #stride_fn(tensor, i as i32) as usize
                    }).collect();

                    #ident {
                        forget: false,
                        storage: Some(storage.forget()),
                        tensor: tensor,
                        size: self.size.to_owned(),
                        stride: stride
                    }
                }
            }

            unsafe fn new_select(&self, dim: usize, i: usize) -> Self {
                let tensor = #new_select_fn(self.tensor, dim as c_int, i as i64);
                let storage: #store_ty_id = #storage_fn(tensor).into();
                let stride : Vec<usize> = (0..#dim_fn(tensor)).map(|i| {
                    #stride_fn(tensor, i as i32) as usize
                }).collect();

                #ident {
                    forget: false,
                    storage: Some(storage.forget()),
                    tensor: tensor,
                    size: self.size.to_owned(),
                    stride: stride
                }
            }

            unsafe fn new_transpose(&self, dim_1: usize, dim_2: usize) -> Self {
                let tensor = #new_transpose_fn(self.tensor, dim_1 as c_int, dim_2 as c_int);
                let storage: #store_ty_id = #storage_fn(tensor).into();
                let stride : Vec<usize> = (0..#dim_fn(tensor)).map(|i| {
                    #stride_fn(tensor, i as i32) as usize
                }).collect();

                #ident {
                    forget: false,
                    storage: Some(storage.forget()),
                    tensor: tensor,
                    size: self.size.to_owned(),
                    stride: stride
                }
            }

            unsafe fn new_unfold(&self, dim: usize, size: usize, step: usize) -> Self {
                let tensor = #new_unfold_fn(self.tensor, dim as c_int, size as i64, step as i64);
                let storage: #store_ty_id = #storage_fn(tensor).into();
                let dim = #dim_fn(tensor);
                let stride : Vec<usize> = (0..dim).map(|i| {
                    #stride_fn(tensor, i as c_int) as usize
                }).collect();
                let size: Vec<usize> = (0..dim).map(|i| {
                    #size_fn(tensor, i as c_int) as usize
                }).collect();

                #ident {
                    forget: false,
                    storage: Some(storage.forget()),
                    tensor: tensor,
                    size: size,
                    stride: stride
                }
            }

            fn new_with_storage_1d(store: Self::Storage, offset: usize, size: usize, stride: usize) -> Self {
                unsafe {
                    let tensor = #new_with_storage_1d_fn(store.storage(), offset, size, stride);
                    #ident {
                        forget: false,
                        storage: Some(store),
                        tensor: tensor,
                        size: vec![size],
                        stride: vec![stride]
                    }
                }
            }

            fn new_with_storage_2d(store: Self::Storage, offset: usize, size: [usize; 2], stride: [usize; 2]) -> Self {
                unsafe {
                    let tensor = #new_with_storage_2d_fn(store.storage(), offset, size[0], stride[0], size[1], stride[1]);
                    #ident {
                        forget: false,
                        storage: Some(store),
                        tensor: tensor,
                        size: size.to_vec(),
                        stride: stride.to_vec()
                    }
                }
            }

            fn new_with_storage_3d(store: Self::Storage, offset: usize, size: [usize; 3], stride: [usize; 3]) -> Self {
                unsafe {
                    let tensor = #new_with_storage_3d_fn(store.storage(), offset, size[0], stride[0], size[1], stride[1], size[2], stride[2]);
                    #ident {
                        forget: false,
                        storage: Some(store),
                        tensor: tensor, 
                        size: size.to_vec(),
                        stride: stride.to_vec()
                    }
                }
            }

            fn new_with_storage_4d(store: Self::Storage, offset: usize, size: [usize; 4], stride: [usize; 4]) -> Self {
                unsafe {
                    let tensor = #new_with_storage_4d_fn(store.storage(), offset, size[0], stride[0], size[1], stride[1], size[2], stride[2], size[3], stride[3]);
                    #ident {
                        forget: false,
                        storage: Some(store),
                        tensor: tensor,
                        size: size.to_vec(),
                        stride: stride.to_vec()
                    }
                }
            }

            fn new_with_size_1d(size: usize) -> Self {
                unsafe {
                    let tensor = #new_with_size_1d_fn(size as i64);
                    let stride = #stride_fn(tensor, 0 as c_int) as usize;

                    let storage: #store_ty_id = #storage_fn(tensor).into();
                    
                    // storage memory in this mode is managed by Caffe2. We need to forget it or it'll be freed twice.

                    #ident {
                        forget: false,
                        storage: Some(storage.forget()),
                        tensor: tensor,
                        size: vec![size],
                        stride: vec![stride]
                    }
                }
            }

            fn new_with_size_2d(size: [usize; 2]) -> Self {
                unsafe {
                    let tensor = #new_with_size_2d_fn(size[0] as i64, size[1] as i64);
                    let stride = [
                        #stride_fn(tensor, 0 as c_int) as usize,
                        #stride_fn(tensor, 1 as c_int) as usize
                    ];
                    let storage: #store_ty_id = #storage_fn(tensor).into();
                    
                    // storage memory in this mode is managed by Caffe2. We need to forget it or it'll be freed twice.
                    #ident {
                        forget: false,
                        storage: Some(storage.forget()),
                        tensor: tensor,
                        size: size.to_vec(),
                        stride: stride.to_vec()
                    }
                }
            }

            fn new_with_size_3d(size: [usize; 3]) -> Self {
                unsafe {
                    let tensor = #new_with_size_3d_fn(size[0] as i64, size[1] as i64, size[2] as i64);
                    let stride = [
                        #stride_fn(tensor, 0 as c_int) as usize,
                        #stride_fn(tensor, 1 as c_int) as usize,
                        #stride_fn(tensor, 2 as c_int) as usize
                    ];
                    let storage: #store_ty_id = #storage_fn(tensor).into();
                    
                    // storage memory in this mode is managed by Caffe2. We need to forget it or it'll be freed twice.
                    #ident {
                        forget: false,
                        storage: Some(storage.forget()),
                        tensor: tensor,
                        size: size.to_vec(),
                        stride: stride.to_vec()
                    }
                }
            }

            fn new_with_size_4d(size: [usize; 4]) -> Self {
                unsafe {
                    let tensor = #new_with_size_4d_fn(size[0] as i64, size[1] as i64, size[2] as i64, size[3] as i64);
                    let stride = [
                        #stride_fn(tensor, 0 as c_int) as usize,
                        #stride_fn(tensor, 1 as c_int) as usize,
                        #stride_fn(tensor, 2 as c_int) as usize,
                        #stride_fn(tensor, 3 as c_int) as usize
                    ];
                    let storage: #store_ty_id = #storage_fn(tensor).into();
                    
                    // storage memory in this mode is managed by Caffe2. We need to forget it or it'll be freed twice.
                    #ident {
                        forget: false,
                        storage: Some(storage.forget()),
                        tensor: tensor,
                        size: size.to_vec(),
                        stride: stride.to_vec()
                    }
                }
            }

            fn data(&mut self) -> &mut [#t] {
                unsafe {
                    std::slice::from_raw_parts_mut(#data_fn(self.tensor), self.storage.as_ref().unwrap().size())
                }
            }

            fn desc(&self) -> String {
                unsafe {
                    #desc_fn(self.tensor).to_string()
                }
            }

            fn dimensions(&self) -> usize {
                unsafe {
                    #dim_fn(self.tensor) as usize
                }
            }

            fn get_0d(&self) -> #t {
                unsafe {
                    #get_0d_fn(self.tensor)
                }
            }

            fn get_1d(&self, i: usize) -> #t {
                unsafe {
                    #get_1d_fn(self.tensor, i as i64)
                }
            }

            fn get_2d(&self, i: [usize; 2]) -> #t {
                unsafe {
                    #get_2d_fn(self.tensor, i[0] as i64, i[1] as i64)
                }
            }

            fn get_3d(&self, i: [usize; 3]) -> #t {
                unsafe {
                    #get_3d_fn(self.tensor, i[0] as i64, i[1] as i64, i[2] as i64)
                }
            }

            fn get_4d(&self, i: [usize; 4]) -> #t {
                unsafe {
                    #get_4d_fn(self.tensor, i[0] as i64, i[1] as i64, i[2] as i64, i[3] as i64)
                }
            }

            fn is_contiguous(&self) -> bool {
                unsafe {
                    #is_contiguous_fn(self.tensor) != 0
                }
            }

            fn numel(&self) -> usize {
                unsafe {
                    #numel_fn(self.tensor) as usize
                }
            }

            fn resize_0d(&mut self) {
                unsafe {
                    self.size.clear();
                    self.stride.clear();
                    #resize_0d_fn(self.tensor);
                }
            }

            fn resize_1d(&mut self, size: usize) {
                unsafe {
                    self.size = vec![size];
                    #resize_1d_fn(self.tensor, size as i64);
                    self.stride = vec![#stride_fn(self.tensor, 0 as c_int) as usize];
                }
            }

            fn resize_2d(&mut self, size: [usize; 2]) {
                unsafe {
                    self.size = size.to_vec();
                    #resize_2d_fn(self.tensor, size[0] as i64, size[1] as i64);
                    self.stride = [
                        #stride_fn(self.tensor, 0 as c_int) as usize,
                        #stride_fn(self.tensor, 1 as c_int) as usize
                    ].to_vec();
                }
            }

            fn resize_3d(&mut self, size: [usize; 3]) {
                unsafe {
                    self.size = size.to_vec();
                    #resize_3d_fn(self.tensor, size[0] as i64, size[1] as i64, size[2] as i64);
                    self.stride = [
                        #stride_fn(self.tensor, 0 as c_int) as usize,
                        #stride_fn(self.tensor, 1 as c_int) as usize,
                        #stride_fn(self.tensor, 2 as c_int) as usize
                    ].to_vec();
                }
            }

            // fn resize_4d(&mut self, size: [usize; 4]) {
            //     unsafe {
            //         self.size = size.to_vec();
            //         #resize_4d_fn(self.tensor, size[0] as i64, size[1] as i64, size[2] as i64, size[3] as i64);

            //         self.stride = [
            //             #stride_fn(self.tensor, 0 as c_int) as usize,
            //             #stride_fn(self.tensor, 1 as c_int) as usize,
            //             #stride_fn(self.tensor, 2 as c_int) as usize,
            //             #stride_fn(self.tensor, 3 as c_int) as usize
            //         ].to_vec();
            //     }
            // }

            // fn resize_5d(&mut self, size: [usize; 5]) {
            //     unsafe {
            //         println!("before");
            //         self.size = size.to_vec();
            //         #resize_5d_fn(self.tensor, size[0] as i64, size[1] as i64, size[2] as i64, size[3] as i64, size[4] as i64);
            //         println!("after; dim={}", #dim_fn(self.tensor));
            //         // self.stride = [
            //         //     #stride_fn(self.tensor, 0 as c_int) as usize,
            //         //     #stride_fn(self.tensor, 1 as c_int) as usize,
            //         //     #stride_fn(self.tensor, 2 as c_int) as usize,
            //         //     #stride_fn(self.tensor, 3 as c_int) as usize,
            //         //     #stride_fn(self.tensor, 4 as c_int) as usize
            //         // ].to_vec();
            //     }
            // }

            fn resize_nd(&mut self, dim: usize, size: &[usize], stride: &[usize]) {
                assert!(dim == size.len() && dim == stride.len(), format!("Size and stride must have exactly {} elements", dim));
                unsafe {
                    self.size = size.to_owned();
                    self.stride = stride.to_owned();
                    #resize_nd_fn(self.tensor, dim as c_int, size.as_ptr() as *const i64, stride.as_ptr() as *const i64);
                }
            }

            fn set_0d(&mut self, v: #t) {
                unsafe {
                    #set_0d_fn(self.tensor, v);
                }
            }

            fn set_1d(&mut self, i: usize, v: #t) {
                unsafe {
                    #set_1d_fn(self.tensor, i as i64, v);
                }
            }

            fn set_2d(&mut self, i: [usize; 2], v: #t) {
                unsafe {
                    #set_2d_fn(self.tensor, i[0] as i64, i[1] as i64, v);
                }
            }

            fn set_3d(&mut self, i: [usize; 3], v: #t) {
                unsafe {
                    #set_3d_fn(self.tensor, i[0] as i64, i[1] as i64, i[2] as i64, v);
                }
            }

            fn set_4d(&mut self, i: [usize; 4], v: #t) {
                unsafe {
                    #set_4d_fn(self.tensor, i[0] as i64, i[1] as i64, i[2] as i64, i[3] as i64, v);
                }
            }

            fn shape(&self) -> (&[usize], &[usize]) {
                (self.size.as_slice(), self.stride.as_slice())
            }

            fn size(&self, dim: usize) -> usize {
                assert!(dim < self.size.len());
                unsafe {
                    #size_fn(self.tensor, dim as c_int) as usize
                }
            }

            fn storage(&mut self) -> &mut Option<Self::Storage> {
                &mut self.storage
            }

            fn storage_offset(&self) -> usize {
                unsafe {
                    #storage_offset_fn(self.tensor) as usize
                }
            }

            fn stride(&self, dim: usize) -> usize {
                assert!(dim < self.stride.len());
                unsafe {
                    #stride_fn(self.tensor, dim as c_int) as usize
                }
            }

            fn squeeze(&self) -> Self {
                let mut new_ts = Self::new();
                unsafe {
                    #squeeze_fn(new_ts.tensor, self.tensor);
                    let dim = new_ts.dimensions();
                    for i in 0..dim {
                        new_ts.size.push(#size_fn(new_ts.tensor, i as c_int) as usize);
                        new_ts.stride.push(#stride_fn(new_ts.tensor, i as c_int) as usize);
                    }
                    new_ts
                }
            }
        }

        impl Clone for #ident {
            fn clone(&self) -> Self {
                unsafe {
                    let clone = #new_clone_fn(self.tensor);
                    let storage: #store_ty_id = #storage_fn(clone).into();
                    let stride : Vec<usize> = (0..#dim_fn(clone)).map(|i| {
                        #stride_fn(clone, i as i32) as usize
                    }).collect();

                    #ident {
                        forget: false,
                        storage: Some(storage.forget()),
                        tensor: clone,
                        size: self.size.to_owned(),
                        stride: stride
                    }
                }
            }
        }

        impl Drop for #ident {
            fn drop(&mut self) {
                unsafe {
                    if !self.forget {
                        #free_fn(self.tensor);
                    }
                }
            }
        }

        impl<'a> IntoIterator for &'a #ident {
            type Item = #t;
            type IntoIter = TensorIterator<'a, #t>;

            fn into_iter(self) -> Self::IntoIter {
                let offset = self.storage_offset();
                let l = self.numel();

                match self.storage {
                    Some(ref storage) => {
                        let offset = self.storage_offset();
                        let (size, stride) = self.shape();

                        TensorIterator::new(&storage[offset..(offset + self.numel())], size, stride)
                    },
                    None => {
                        TensorIterator::new(&[], &[0], &[])
                    }
                }
            }
        }

        impl<'a> IntoIterator for &'a mut #ident {
            type Item = &'a mut #t;
            type IntoIter = TensorIterMut<'a, #t>;

            fn into_iter(self) -> Self::IntoIter {
                // need unsafe because we both borrow and borrow mut on self at the same time.
                unsafe {
                    // borrow mut self here
                    let data = &mut *(self.data() as *mut [#t]);
                    // borrow self here
                    let (size, stride) = self.shape();

                    TensorIterMut::new(data, size, stride)
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
