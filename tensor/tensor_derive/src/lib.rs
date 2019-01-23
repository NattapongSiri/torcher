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

        pub struct #ident<'a> {
            pub data: &'a mut [#t],
            forget : bool,
            storage : Option<#store_ty_id<'a>>,
            storage_bound: usize,
            tensor : *mut #c_ty_id,
            size : Vec<usize>,
            stride : Vec<usize>
        }

        impl<'a> #ident<'a> {
            /// Convenience method that return iterator of
            /// current object
            fn iter(&self) -> TensorIterator<#t> {
                self.into_iter()
            }

            /// Convenience method that return mutable iterator of
            /// current object
            fn iter_mut(&mut self) -> TensorIterMut<#t> {
                self.into_iter()
            }

            /// Get short description of storage.
            /// This includes name of storage, size, and
            /// sample data if it has more than 20 elements.
            /// If it has less than 20 elements, it'll display
            /// every elements.
            fn short_desc(&mut self) -> String {
                let size = self.size.as_slice();
                let stride = self.stride.as_slice();
                let data = self.data();
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

        impl<'a> Tensor<#t> for #ident<'a> {
            type Storage = #store_ty_id<'a>;

            fn new() -> #ident<'a> {
                unsafe {
                    #ident {
                        data: &mut [],
                        forget: false,
                        storage: None,
                        storage_bound: 0,
                        tensor: #new_fn(),
                        size: Vec::new(),
                        stride: Vec::new()
                    }
                }
            }

            fn new_contiguous(&self) -> #ident<'a> {
                unsafe {
                    let cont = #new_contiguous_fn(self.tensor);
                    let storage = #store_ty_id::from(#storage_fn(cont)).forget();
                    let data = std::slice::from_raw_parts_mut(#data_fn(cont), storage.size());
                    let stride : Vec<usize> = (0..#dim_fn(cont)).map(|i| {
                        #stride_fn(cont, i as i32) as usize
                    }).collect();

                    #ident {
                        data: data,
                        forget: false,
                        storage: Some(storage),
                        storage_bound: self.storage_bound,
                        tensor: cont,
                        size: self.size.to_owned(),
                        stride: stride
                    }
                }
            }

            unsafe fn new_narrow(&self, dim: usize, i: usize, size: usize) -> #ident<'a> {
                let tensor = #new_narrow_fn(self.tensor, dim as c_int, i as i64, size as i64);
                let storage = #store_ty_id::from(#storage_fn(tensor)).forget();
                let mut size = Vec::new();
                let dim = #dim_fn(tensor);
                let mut big_bound = 0;
                let stride : Vec<usize> = (0..dim).map(|i| {
                    let cur_stride = #stride_fn(tensor, i as i32) as usize;
                    let cur_size = #size_fn(tensor, i as i32) as usize;
                    size.push(cur_size);
                    let cur_bound = cur_stride * cur_size;

                    if cur_bound > big_bound && i < dim - 1 {
                        big_bound = cur_bound;
                    }

                    cur_stride
                }).collect();

                let storage_bound = big_bound + size[size.len() - 1] - 1;
                let data = std::slice::from_raw_parts_mut(#data_fn(tensor), storage_bound);

                #ident {
                    data: data,
                    forget: false,
                    storage: Some(storage),
                    storage_bound: storage_bound,
                    tensor: tensor,
                    size: size,
                    stride: stride
                }
            }

            unsafe fn new_select(&self, dim: usize, i: usize) -> #ident<'a> {
                let tensor = #new_select_fn(self.tensor, dim as c_int, i as i64);
                let storage = #store_ty_id::from(#storage_fn(tensor)).forget();
                let mut size = Vec::new();
                let dim = #dim_fn(tensor);
                let mut big_bound = 0;
                let stride : Vec<usize> = (0..dim).map(|i| {
                    let cur_stride = #stride_fn(tensor, i as i32) as usize;
                    let cur_size = #size_fn(tensor, i as i32) as usize;
                    size.push(cur_size);
                    let cur_bound = cur_stride * cur_size;

                    if cur_bound > big_bound && i < dim - 1 {
                        big_bound = cur_bound;
                    }

                    cur_stride
                }).collect();

                let storage_bound = big_bound + size[size.len() - 1] - 1;
                let data = std::slice::from_raw_parts_mut(#data_fn(tensor), storage_bound);

                #ident {
                    data: data,
                    forget: false,
                    storage: Some(storage),
                    storage_bound: storage_bound,
                    tensor: tensor,
                    size: size,
                    stride: stride
                }
            }

            unsafe fn new_transpose(&self, dim_1: usize, dim_2: usize) -> #ident<'a> {
                let tensor = #new_transpose_fn(self.tensor, dim_1 as c_int, dim_2 as c_int);
                let storage: #store_ty_id = #storage_fn(tensor).into();
                let mut size = Vec::new();
                let dim = #dim_fn(tensor);
                let mut big_bound = 0;
                let stride : Vec<usize> = (0..dim).map(|i| {
                    let cur_stride = #stride_fn(tensor, i as i32) as usize;
                    let cur_size = #size_fn(tensor, i as i32) as usize;
                    size.push(cur_size);
                    let cur_bound = cur_stride * cur_size;

                    if cur_bound > big_bound && i < dim - 1 {
                        big_bound = cur_bound;
                    }

                    cur_stride
                }).collect();

                let storage_bound = big_bound + size[size.len() - 1] - 1;
                let data = std::slice::from_raw_parts_mut(#data_fn(tensor), storage_bound);

                #ident {
                    data: data,
                    forget: false,
                    storage: Some(storage.forget()),
                    storage_bound: storage_bound,
                    tensor: tensor,
                    size: size,
                    stride: stride
                }
            }

            unsafe fn new_unfold(&self, dim: usize, size: usize, step: usize) -> #ident<'a> {
                let tensor = #new_unfold_fn(self.tensor, dim as c_int, size as i64, step as i64);
                let storage: #store_ty_id = #storage_fn(tensor).into();
                let mut size = Vec::new();
                let dim = #dim_fn(tensor);
                let mut big_bound = 0;
                let stride : Vec<usize> = (0..dim).map(|i| {
                    let cur_stride = #stride_fn(tensor, i as i32) as usize;
                    let cur_size = #size_fn(tensor, i as i32) as usize;
                    size.push(cur_size);
                    let cur_bound = cur_stride * cur_size;

                    if cur_bound > big_bound && i < dim - 1 {
                        big_bound = cur_bound;
                    }

                    cur_stride
                }).collect();

                let storage_bound = big_bound + size[size.len() - 1] - 1;
                let data = std::slice::from_raw_parts_mut(#data_fn(tensor), storage_bound);

                #ident {
                    data: data,
                    forget: false,
                    storage: Some(storage.forget()),
                    storage_bound: storage_bound,
                    tensor: tensor,
                    size: size,
                    stride: stride
                }
            }

            fn new_with_storage_1d(store: Self::Storage, offset: usize, size: usize) -> #ident<'a> {
                unsafe {
                    // Caffe2 stride of 1d doesn't matter. It can be set to any value.
                    let tensor = #new_with_storage_1d_fn(store.storage(), offset, size, 1);
                    let data = std::slice::from_raw_parts_mut(#data_fn(tensor), size);

                    #ident {
                        data: data,
                        forget: false,
                        storage: Some(store),
                        storage_bound: offset + size,
                        tensor: tensor,
                        size: vec![size],
                        stride: vec![1]
                    }
                }
            }

            fn new_with_storage_2d(store: Self::Storage, offset: usize, size: [usize; 2], stride: usize) -> #ident<'a> {
                unsafe {
                    // Caffe2 last dim doesn't have stride so it can be set to any value.
                    let tensor = #new_with_storage_2d_fn(store.storage(), offset, size[0], stride, size[1], 1);
                    let dim = #dim_fn(tensor);
                    let mut big_bound = 0;
                    let stride : Vec<usize> = (0..dim).map(|i| {
                        let cur_stride = #stride_fn(tensor, i as i32) as usize;
                        let cur_bound = cur_stride * size[i as usize];

                        if cur_bound > big_bound && i < dim - 1 {
                            big_bound = cur_bound;
                        }

                        cur_stride
                    }).collect();

                    let storage_bound = big_bound + size[size.len() - 1] - 1;
                    let mut stride = stride.to_vec();
                    stride.push(1); // last dim stride for sake of consistency
                    let data = std::slice::from_raw_parts_mut(#data_fn(tensor), storage_bound);
                    
                    #ident {
                        data: data,
                        forget: false,
                        storage: Some(store),
                        storage_bound: storage_bound,
                        tensor: tensor,
                        size: size.to_vec(),
                        stride: stride
                    }
                }
            }

            fn new_with_storage_3d(store: Self::Storage, offset: usize, size: [usize; 3], stride: [usize; 2]) -> #ident<'a> {
                unsafe {
                    // Caffe2 last dim doesn't have stride so it can be set to any value.
                    let tensor = #new_with_storage_3d_fn(store.storage(), offset, size[0], stride[0], size[1], stride[1], size[2], 1);
                    let dim = #dim_fn(tensor);
                    let mut big_bound = 0;
                    let stride : Vec<usize> = (0..dim).map(|i| {
                        let cur_stride = #stride_fn(tensor, i as i32) as usize;
                        let cur_bound = cur_stride * size[i as usize];

                        if cur_bound > big_bound && i < dim - 1 {
                            big_bound = cur_bound;
                        }

                        cur_stride
                    }).collect();

                    let storage_bound = big_bound + size[size.len() - 1] - 1;
                    let mut stride = stride.to_vec();
                    stride.push(1); // last dim stride for sake of consistency
                    let data = std::slice::from_raw_parts_mut(#data_fn(tensor), storage_bound);

                    #ident {
                        data: data,
                        forget: false,
                        storage: Some(store),
                        storage_bound: storage_bound,
                        tensor: tensor, 
                        size: size.to_vec(),
                        stride: stride
                    }
                }
            }

            fn new_with_storage_4d(store: Self::Storage, offset: usize, size: [usize; 4], stride: [usize; 3]) -> #ident<'a> {
                unsafe {
                    // Caffe2 last dim doesn't have stride so it can be set to any value.
                    let tensor = #new_with_storage_4d_fn(store.storage(), offset, size[0], stride[0], size[1], stride[1], size[2], stride[2], size[3], 1);
                    let dim = #dim_fn(tensor);
                    let mut big_bound = 0;
                    let stride : Vec<usize> = (0..dim).map(|i| {
                        let cur_stride = #stride_fn(tensor, i as i32) as usize;
                        let cur_bound = cur_stride * size[i as usize];

                        if cur_bound > big_bound && i < dim - 1 {
                            big_bound = cur_bound;
                        }

                        cur_stride
                    }).collect();

                    let storage_bound = big_bound + size[size.len() - 1] - 1;
                    let mut stride = stride.to_vec();
                    stride.push(1); // last dim stride for sake of consistency
                    let storage = #store_ty_id::from(#storage_fn(tensor)).forget();
                    let data = std::slice::from_raw_parts_mut(#data_fn(tensor), storage_bound);

                    #ident {
                        data: data,
                        forget: false,
                        storage: Some(storage),
                        storage_bound: storage_bound,
                        tensor: tensor,
                        size: size.to_vec(),
                        stride: stride
                    }
                }
            }

            fn new_with_size_1d(size: usize) -> #ident<'a> {
                unsafe {
                    let tensor = #new_with_size_1d_fn(size as i64);
                    let stride = #stride_fn(tensor, 0 as c_int) as usize;

                    let storage = #store_ty_id::from(#storage_fn(tensor)).forget();
                    let data = std::slice::from_raw_parts_mut(#data_fn(tensor), size);
                    
                    // storage memory in this mode is managed by Caffe2. We need to forget it or it'll be freed twice.

                    #ident {
                        data: data,
                        forget: false,
                        storage: Some(storage),
                        storage_bound: size,
                        tensor: tensor,
                        size: vec![size],
                        stride: vec![stride]
                    }
                }
            }

            fn new_with_size_2d(size: [usize; 2]) -> #ident<'a> {
                unsafe {
                    let tensor = #new_with_size_2d_fn(size[0] as i64, size[1] as i64);
                    let stride = [
                        #stride_fn(tensor, 0 as c_int) as usize,
                        #stride_fn(tensor, 1 as c_int) as usize
                    ];
                    let storage: #store_ty_id = #storage_fn(tensor).into();
                    let dim = #dim_fn(tensor);
                    let mut big_bound = 0;
                    let stride : Vec<usize> = (0..dim).map(|i| {
                        let cur_stride = #stride_fn(tensor, i as i32) as usize;
                        let cur_bound = cur_stride * size[i as usize];

                        if cur_bound > big_bound && i < dim - 1 {
                            big_bound = cur_bound;
                        }

                        cur_stride
                    }).collect();

                    let storage_bound = big_bound + size[size.len() - 1] - 1;
                    let data = std::slice::from_raw_parts_mut(#data_fn(tensor), storage_bound);
                    
                    // storage memory in this mode is managed by Caffe2. We need to forget it or it'll be freed twice.
                    #ident {
                        data: data,
                        forget: false,
                        storage: Some(storage.forget()),
                        storage_bound: storage_bound,
                        tensor: tensor,
                        size: size.to_vec(),
                        stride: stride.to_vec()
                    }
                }
            }

            fn new_with_size_3d(size: [usize; 3]) -> #ident<'a> {
                unsafe {
                    let tensor = #new_with_size_3d_fn(size[0] as i64, size[1] as i64, size[2] as i64);
                    let stride = [
                        #stride_fn(tensor, 0 as c_int) as usize,
                        #stride_fn(tensor, 1 as c_int) as usize,
                        #stride_fn(tensor, 2 as c_int) as usize
                    ];
                    let storage = #store_ty_id::from(#storage_fn(tensor)).forget();
                    let dim = #dim_fn(tensor);
                    let mut big_bound = 0;
                    let stride : Vec<usize> = (0..dim).map(|i| {
                        let cur_stride = #stride_fn(tensor, i as i32) as usize;
                        let cur_bound = cur_stride * size[i as usize];

                        if cur_bound > big_bound && i < dim - 1 {
                            big_bound = cur_bound;
                        }

                        cur_stride
                    }).collect();

                    let storage_bound = big_bound + size[size.len() - 1] - 1;
                    let data = std::slice::from_raw_parts_mut(#data_fn(tensor), storage_bound);
                    
                    // storage memory in this mode is managed by Caffe2. We need to forget it or it'll be freed twice.
                    #ident {
                        data: data,
                        forget: false,
                        storage: Some(storage),
                        storage_bound: storage_bound,
                        tensor: tensor,
                        size: size.to_vec(),
                        stride: stride.to_vec()
                    }
                }
            }

            fn new_with_size_4d(size: [usize; 4]) -> #ident<'a> {
                unsafe {
                    let tensor = #new_with_size_4d_fn(size[0] as i64, size[1] as i64, size[2] as i64, size[3] as i64);
                    let stride = [
                        #stride_fn(tensor, 0 as c_int) as usize,
                        #stride_fn(tensor, 1 as c_int) as usize,
                        #stride_fn(tensor, 2 as c_int) as usize,
                        #stride_fn(tensor, 3 as c_int) as usize
                    ];
                    let storage = #store_ty_id::from(#storage_fn(tensor)).forget();
                    let dim = #dim_fn(tensor);
                    let mut big_bound = 0;
                    let stride : Vec<usize> = (0..dim).map(|i| {
                        let cur_stride = #stride_fn(tensor, i as i32) as usize;
                        let cur_bound = cur_stride * size[i as usize];

                        if cur_bound > big_bound && i < dim - 1 {
                            big_bound = cur_bound;
                        }

                        cur_stride
                    }).collect();

                    let storage_bound = big_bound + size[size.len() - 1] - 1;
                    let data = std::slice::from_raw_parts_mut(#data_fn(tensor), storage_bound);
                    
                    // storage memory in this mode is managed by Caffe2. We need to forget it or it'll be freed twice.
                    #ident {
                        data: data,
                        forget: false,
                        storage: Some(storage),
                        storage_bound: storage_bound,
                        tensor: tensor,
                        size: size.to_vec(),
                        stride: stride.to_vec()
                    }
                }
            }

            fn data(&self) -> &[#t] {
                self.data
            }

            fn data_mut(&mut self) -> &mut [#t] {
                self.data
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
                    self.storage_bound = 1;
                    self.data = std::slice::from_raw_parts_mut(#data_fn(self.tensor), 1);
                    #resize_0d_fn(self.tensor);
                }
            }

            fn resize_1d(&mut self, size: usize) {
                unsafe {
                    self.size = vec![size];
                    self.stride = vec![1];
                    self.storage_bound = size;
                    #resize_1d_fn(self.tensor, size as i64);
                    self.stride = vec![#stride_fn(self.tensor, 0 as c_int) as usize];
                    let data = std::slice::from_raw_parts_mut(#data_fn(self.tensor), size);
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
                    
                    let storage_bound = self.stride[0] * self.size[0] + self.size[1] - 1;
                    self.storage_bound = storage_bound;
                    let data = std::slice::from_raw_parts_mut(#data_fn(self.tensor), storage_bound);
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

                    let mut storage_bound = 0;

                    for i in 0..(self.stride.len() - 1) {
                        let cur_bound = self.stride[i] * size[i];
                        if cur_bound > storage_bound {
                            storage_bound = cur_bound;
                        }
                    }

                    storage_bound += size[size.len() - 1] - 1;
                    self.storage_bound = storage_bound;
                    let data = std::slice::from_raw_parts_mut(#data_fn(self.tensor), storage_bound);
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
                assert_eq!(dim, size.len(), "Size must have exactly {} elements", dim);
                assert_eq!(dim - 1, stride.len(), "Stride must have exactly {} elements", dim - 1);
                unsafe {
                    self.size = size.to_owned();
                    self.stride = stride.to_owned();
                    self.stride.push(1); // just to conform to Caffe2 API
                    #resize_nd_fn(self.tensor, dim as c_int, size.as_ptr() as *const i64, self.stride.as_ptr() as *const i64);

                    let mut storage_bound = 0;

                    for i in 0..(self.size.len() - 1) {
                        let cur_bound = self.stride[i] * size[i];

                        if cur_bound > storage_bound {
                            storage_bound = cur_bound;
                        }
                    }

                    storage_bound += size[size.len() - 1] - 1;
                    self.storage_bound = storage_bound;
                    let data = std::slice::from_raw_parts_mut(#data_fn(self.tensor), storage_bound);
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
                    let dim = #dim_fn(new_ts.tensor);
                    let mut storage_bound = 0;

                    for i in 0..(dim - 1) {
                        new_ts.size.push(#size_fn(new_ts.tensor, i as c_int) as usize);
                        new_ts.stride.push(#stride_fn(new_ts.tensor, i as c_int) as usize);
                        let cur_bound = new_ts.size[i as usize] * new_ts.stride[i as usize];

                        if cur_bound > storage_bound {
                            storage_bound = cur_bound;
                        }
                    }

                    new_ts.size.push(#size_fn(new_ts.tensor, dim - 1 as c_int) as usize);
                    new_ts.stride.push(#stride_fn(new_ts.tensor, dim - 1 as c_int) as usize);
                    storage_bound += new_ts.size[dim as usize - 1];
                    new_ts.storage_bound = storage_bound;
                    new_ts.data = std::slice::from_raw_parts_mut(#data_fn(new_ts.tensor), storage_bound);

                    new_ts
                }
            }
        }

        impl<'a> Clone for #ident<'a> {
            fn clone(&self) -> #ident<'a> {
                unsafe {
                    let clone = #new_clone_fn(self.tensor);
                    let storage = #store_ty_id::from(#storage_fn(clone)).forget();
                    let stride : Vec<usize> = (0..#dim_fn(clone)).map(|i| {
                        #stride_fn(clone, i as i32) as usize
                    }).collect();

                    #ident {
                        data: std::slice::from_raw_parts_mut(#data_fn(self.tensor), self.storage_bound),
                        forget: false,
                        storage: Some(storage),
                        storage_bound: self.storage_bound,
                        tensor: clone,
                        size: self.size.to_owned(),
                        stride: stride
                    }
                }
            }
        }

        /// It allow read only access to underlying data.
        /// It's similar to calling to [data](trait.Tensor.html#tymethod.data) function
        /// but that function need mut access.
        impl<'a> Deref for #ident<'a> {
            type Target=[#t];

            fn deref(&self) -> &Self::Target {
                self.data()
            }
        }

        /// It deref into underlying storage data.
        /// It's similar to calling to [data](trait.Tensor.html#tymethod.data) function
        impl<'a> DerefMut for #ident<'a> {
            fn deref_mut(&mut self) -> &mut Self::Target {
                self.data_mut()
            }
        }

        impl<'a> Drop for #ident<'a> {
            fn drop(&mut self) {
                unsafe {
                    if !self.forget {
                        #free_fn(self.tensor);
                    }
                }
            }
        }

        impl<'a, 'b: 'a> IntoIterator for &'a #ident<'b> {
            type Item = #t;
            type IntoIter = TensorIterator<'a, #t>;

            fn into_iter(self) -> Self::IntoIter {
                let (size, stride) = self.shape();
                TensorIterator::new(self.data(), size, stride)
            }
        }

        impl<'a, 'b: 'a> IntoIterator for &'a mut #ident<'b> {
            type Item = &'a mut #t;
            type IntoIter = TensorIterMut<'a, #t>;

            fn into_iter(self) -> Self::IntoIter {
                // need unsafe because we both borrow and borrow mut on self at the same time.
                unsafe {
                    // borrow mut self here
                    let data = &mut *(self.data_mut() as *mut [#t]);
                    // borrow self here
                    let (size, stride) = self.shape();

                    TensorIterMut::new(data, size, stride)
                }
            }
        }
    };

    TokenStream::from(expanded)
}