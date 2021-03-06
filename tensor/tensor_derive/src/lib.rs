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

    let cat_fn = _make_ident(&c_ty, "_catArray");
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
    let storage_nd_fn = _make_ident(&c_ty, "_setStorageNd");
    let storage_fn = _make_ident(&c_ty, "_storage");
    let storage_offset_fn = _make_ident(&c_ty, "_storageOffset");
    let stride_fn = _make_ident(&c_ty, "_stride");
    let squeeze_fn = _make_ident(&c_ty, "_squeeze");
    let squeeze_1d_fn = _make_ident(&c_ty, "_squeeze1d");

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

            fn #cat_fn(result: *mut #c_ty_id, inputs: *const*const #c_ty_id, input_len: i64, dim: i64);
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
            fn #storage_nd_fn(tensor: *const #c_ty_id, storage: #c_storage_ty_id, offset: i64, dim: c_int, size: *const i64, stride: *const i64);
            fn #storage_fn(tensor: *mut #c_ty_id) -> *mut #c_storage_ty_id;
            fn #storage_offset_fn(tensor: *const #c_ty_id) -> i64;
            fn #stride_fn(tensor: *const #c_ty_id, dim: c_int) -> i64;
            fn #squeeze_fn(tensor: *const #c_ty_id, res: *mut #c_ty_id);
            fn #squeeze_1d_fn(tensor: *const #c_ty_id, res: *mut #c_ty_id, dim: i64);
        }

        pub struct #ident {
            data: *mut [#t],
            forget : bool,
            storage : Option<#store_ty_id>,
            storage_bound: usize,
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

        impl CreateOp<#store_ty_id> for #ident {
            type Datum = #t;

            fn new() -> #ident {
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

            fn new_concat(&self, tensors: &[Self], dim: usize) -> Self {
                unsafe {
                    // construct *const *const #c_ty_id. A pointer to pointer to tensor
                    let mut buffer = Vec::new();
                    buffer.push(self.tensor as *const #c_ty_id);
                    tensors.iter().for_each(|ts| buffer.push(ts.tensor as *const #c_ty_id));
                    let mut new_ts = Self::new(); // create new contiguous tensor
                    
                    #cat_fn(
                        new_ts.tensor, 
                        buffer.as_ptr(),
                        buffer.len() as i64,
                        dim as i64
                    );

                    // update size and stride
                    let last_dim = self.shape().0.len();
                    let mut sizes = Vec::with_capacity(dim);
                    let mut strides = Vec::with_capacity(dim);
                    (0..last_dim).for_each(|i| {
                        sizes.push(#size_fn(new_ts.tensor, i as i32) as usize);
                        strides.push(#stride_fn(new_ts.tensor, i as i32) as usize);
                    });
                    new_ts.size = sizes;
                    new_ts.stride = strides;
                    
                    // perform data map on new tensor
                    new_ts.storage = Some(#store_ty_id::from(#storage_fn(new_ts.tensor)).forget());
                    let storage_bound = new_ts.size[0] * new_ts.stride[0];
                    new_ts.data = std::slice::from_raw_parts_mut(#data_fn(new_ts.tensor), storage_bound) as *mut [#t];

                    new_ts
                }
            }

            fn new_contiguous(&self) -> #ident {
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

            unsafe fn new_narrow(&self, dim: usize, i: usize, size: usize) -> #ident {
                let tensor = #new_narrow_fn(self.tensor, dim as c_int, i as i64, size as i64);
                let storage = #store_ty_id::from(#storage_fn(tensor)).forget();
                let mut sizes = self.size.to_owned();
                let old_size = sizes.remove(dim);
                sizes.insert(dim, size);
                let storage_bound = self.storage_bound - (old_size - size) * self.stride[dim];
                let data = std::slice::from_raw_parts_mut(#data_fn(tensor), storage_bound);
                
                #ident {
                    data: data,
                    forget: false,
                    storage: Some(storage),
                    storage_bound: storage_bound,
                    tensor: tensor,
                    size: sizes,
                    stride: self.stride.to_owned()
                }
            }

            unsafe fn new_select(&self, dim: usize, i: usize) -> #ident {
                let tensor = #new_select_fn(self.tensor, dim as c_int, i as i64);
                let storage = #store_ty_id::from(#storage_fn(tensor)).forget();
                let mut size = Vec::new();
                let dim = #dim_fn(tensor) as usize;
                let mut big_bound = 0;
                let mut unit_size = 1;
                let stride : Vec<usize> = (0usize..dim).map(|i| {
                    let cur_stride = #stride_fn(tensor, i as i32) as usize;
                    let cur_size = #size_fn(tensor, i as i32) as usize;
                    size.push(cur_size);
                    let cur_bound = cur_stride * size[i];

                    if cur_bound > big_bound && i < dim - 1 {
                        big_bound = cur_bound;
                    }

                    if cur_stride == 1 {
                        unit_size = size[i];
                    }

                    cur_stride
                }).collect();

                let storage_bound = match big_bound {
                    0 => unit_size,
                    n => big_bound
                };
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

            unsafe fn new_transpose(&self, dim_1: usize, dim_2: usize) -> #ident {
                let tensor = #new_transpose_fn(self.tensor, dim_1 as c_int, dim_2 as c_int);
                let storage: #store_ty_id = #storage_fn(tensor).into();
                let mut size = Vec::new();
                let dim = #dim_fn(tensor) as usize;

                let stride : Vec<usize> = (0usize..dim).map(|i| {
                    let cur_stride = #stride_fn(tensor, i as i32) as usize;
                    let cur_size = #size_fn(tensor, i as i32) as usize;
                    size.push(cur_size);

                    cur_stride
                }).collect();
                let data = std::slice::from_raw_parts_mut(#data_fn(tensor), self.storage_bound);

                #ident {
                    data: data,
                    forget: false,
                    storage: Some(storage.forget()),
                    storage_bound: self.storage_bound,
                    tensor: tensor,
                    size: size,
                    stride: stride
                }
            }

            unsafe fn new_unfold(&self, dim: usize, size: usize, step: usize) -> #ident {
                let tensor = #new_unfold_fn(self.tensor, dim as c_int, size as i64, step as i64);
                let storage: #store_ty_id = #storage_fn(tensor).into();
                let mut size = Vec::new();
                let dim = #dim_fn(tensor) as usize;
                let stride : Vec<usize> = (0usize..dim).map(|i| {
                    let cur_stride = #stride_fn(tensor, i as i32) as usize;
                    let cur_size = #size_fn(tensor, i as i32) as usize;
                    size.push(cur_size);
                    cur_stride
                }).collect();
                let data = std::slice::from_raw_parts_mut(#data_fn(tensor), self.storage_bound);

                #ident {
                    data: data,
                    forget: false,
                    storage: Some(storage.forget()),
                    storage_bound: self.storage_bound,
                    tensor: tensor,
                    size: size,
                    stride: stride
                }
            }

            fn new_with_storage_1d(store: #store_ty_id, offset: usize, size: usize) -> #ident {
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

            fn new_with_storage_2d(store: #store_ty_id, offset: usize, size: [usize; 2], stride: usize) -> #ident {
                unsafe {
                    // Caffe2 last dim doesn't have stride so it can be set to any value.
                    let tensor = #new_with_storage_2d_fn(store.storage(), offset, size[0], stride, size[1], 1);
                    let dim = #dim_fn(tensor) as usize;
                    let mut big_bound = 0;
                    let mut unit_size = 1;
                    let stride : Vec<usize> = (0usize..dim).map(|i| {
                        let cur_stride = #stride_fn(tensor, i as i32) as usize;
                        let cur_bound = cur_stride * size[i];

                        if cur_bound > big_bound && i < dim - 1 {
                            big_bound = cur_bound;
                        }

                        if cur_stride == 1 {
                            unit_size = size[i];
                        }

                        cur_stride
                    }).collect();

                    let storage_bound = match big_bound {
                        0 => unit_size,
                        n => big_bound
                    };
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

            fn new_with_storage_3d(store: #store_ty_id, offset: usize, size: [usize; 3], stride: [usize; 2]) -> #ident {
                unsafe {
                    // Caffe2 last dim doesn't have stride so it can be set to any value.
                    let tensor = #new_with_storage_3d_fn(store.storage(), offset, size[0], stride[0], size[1], stride[1], size[2], 1);
                    let dim = #dim_fn(tensor) as usize;
                    let mut big_bound = 0;
                    let mut unit_size = 1;
                    let stride : Vec<usize> = (0usize..dim).map(|i| {
                        let cur_stride = #stride_fn(tensor, i as i32) as usize;
                        let cur_bound = cur_stride * size[i];

                        if cur_bound > big_bound && i < dim - 1 {
                            big_bound = cur_bound;
                        }

                        if cur_stride == 1 {
                            unit_size = size[i];
                        }

                        cur_stride
                    }).collect();

                    let storage_bound = match big_bound {
                        0 => unit_size,
                        n => big_bound
                    };
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

            fn new_with_storage_4d(store: #store_ty_id, offset: usize, size: [usize; 4], stride: [usize; 3]) -> #ident {
                unsafe {
                    // Caffe2 last dim doesn't have stride so it can be set to any value.
                    let tensor = #new_with_storage_4d_fn(store.storage(), offset, size[0], stride[0], size[1], stride[1], size[2], stride[2], size[3], 1);
                    let dim = #dim_fn(tensor) as usize;
                    let mut big_bound = 0;
                    let mut unit_size = 1;
                    let stride : Vec<usize> = (0usize..dim).map(|i| {
                        let cur_stride = #stride_fn(tensor, i as i32) as usize;
                        let cur_bound = cur_stride * size[i];

                        if cur_bound > big_bound && i < dim - 1 {
                            big_bound = cur_bound;
                        }

                        if cur_stride == 1 {
                            unit_size = size[i];
                        }

                        cur_stride
                    }).collect();

                    let storage_bound = match big_bound {
                        0 => unit_size,
                        n => big_bound
                    };
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

            fn new_with_storage_nd(store: #store_ty_id, offset: usize, size: &[usize], stride: &[usize]) -> #ident {
                assert!(size.len() == stride.len() - 1 || size.len() == stride.len(), "Stride shall have either n - 1 elements or n elements where n = size.len()");
                
                let mut storage_len = 0;

                for i in 0..(stride.len() - 1) {
                    let cur_len = stride[i] * size[i];
                    if cur_len > storage_len {
                        storage_len = cur_len;
                    }
                }

                storage_len += match storage_len {
                    0 => size[size.len() - 1],
                    n => size[size.len() - 1] - 1
                };
                let mut stride = stride.to_owned();

                if stride.len() == size.len() - 1 {
                    stride.push(1);
                }

                unsafe {
                    // Caffe2 last dim doesn't have stride so it can be set to any value.
                    let tensor = #new_with_storage_1d_fn(store.storage(), offset, storage_len, 1);
                    #resize_nd_fn(tensor, size.len() as i32, size.as_ptr() as *const i64, stride.as_ptr() as *const i64);

                    let dim = #dim_fn(tensor) as usize;
                    let mut big_bound = 0;
                    let mut unit_size = 1;
                    let stride : Vec<usize> = (0usize..dim).map(|i| {
                        let cur_stride = #stride_fn(tensor, i as i32) as usize;
                        let cur_bound = cur_stride * size[i];

                        if cur_bound > big_bound && i < dim - 1 {
                            big_bound = cur_bound;
                        }

                        if cur_stride == 1 {
                            unit_size = size[i];
                        }

                        cur_stride
                    }).collect();

                    let storage_bound = match big_bound {
                        0 => unit_size,
                        n => big_bound
                    };
                    let mut stride = stride.to_vec();
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

            fn new_with_size_1d(size: usize) -> #ident {
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

            fn new_with_size_2d(size: [usize; 2]) -> #ident {
                unsafe {
                    let tensor = #new_with_size_2d_fn(size[0] as i64, size[1] as i64);
                    let stride = [
                        #stride_fn(tensor, 0 as c_int) as usize,
                        #stride_fn(tensor, 1 as c_int) as usize
                    ];
                    let storage: #store_ty_id = #storage_fn(tensor).into();
                    let dim = #dim_fn(tensor) as usize;
                    let mut big_bound = 0;
                    let mut unit_size = 1;
                    let stride : Vec<usize> = (0usize..dim).map(|i| {
                        let cur_stride = #stride_fn(tensor, i as i32) as usize;
                        let cur_bound = cur_stride * size[i];

                        if cur_bound > big_bound && i < dim - 1 {
                            big_bound = cur_bound;
                        }

                        if cur_stride == 1 {
                            unit_size = size[i];
                        }

                        cur_stride
                    }).collect();

                    let storage_bound = match big_bound {
                        0 => unit_size,
                        n => big_bound
                    };
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

            fn new_with_size_3d(size: [usize; 3]) -> #ident {
                unsafe {
                    let tensor = #new_with_size_3d_fn(size[0] as i64, size[1] as i64, size[2] as i64);
                    let stride = [
                        #stride_fn(tensor, 0 as c_int) as usize,
                        #stride_fn(tensor, 1 as c_int) as usize,
                        #stride_fn(tensor, 2 as c_int) as usize
                    ];
                    let storage = #store_ty_id::from(#storage_fn(tensor)).forget();
                    let dim = #dim_fn(tensor) as usize;
                    let mut big_bound = 0;
                    let mut unit_size = 1;
                    let stride : Vec<usize> = (0usize..dim).map(|i| {
                        let cur_stride = #stride_fn(tensor, i as i32) as usize;
                        let cur_bound = cur_stride * size[i];

                        if cur_bound > big_bound && i < dim - 1 {
                            big_bound = cur_bound;
                        }

                        if cur_stride == 1 {
                            unit_size = size[i];
                        }

                        cur_stride
                    }).collect();

                    let storage_bound = match big_bound {
                        0 => unit_size,
                        n => big_bound
                    };
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

            fn new_with_size_4d(size: [usize; 4]) -> #ident {
                unsafe {
                    let tensor = #new_with_size_4d_fn(size[0] as i64, size[1] as i64, size[2] as i64, size[3] as i64);
                    let stride = [
                        #stride_fn(tensor, 0 as c_int) as usize,
                        #stride_fn(tensor, 1 as c_int) as usize,
                        #stride_fn(tensor, 2 as c_int) as usize,
                        #stride_fn(tensor, 3 as c_int) as usize
                    ];
                    let storage = #store_ty_id::from(#storage_fn(tensor)).forget();
                    let dim = #dim_fn(tensor) as usize;
                    let mut big_bound = 0;
                    let mut unit_size = 1;
                    let stride : Vec<usize> = (0usize..dim).map(|i| {
                        let cur_stride = #stride_fn(tensor, i as i32) as usize;
                        let cur_bound = cur_stride * size[i];

                        if cur_bound > big_bound && i < dim - 1 {
                            big_bound = cur_bound;
                        }

                        if cur_stride == 1 {
                            unit_size = size[i];
                        }

                        cur_stride
                    }).collect();

                    let storage_bound = match big_bound {
                        0 => unit_size,
                        n => big_bound
                    };
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

            unsafe fn forget(mut self) -> Self {
                self.forget = true;

                self
            }
        }

        impl BasicManipulateOp<#store_ty_id> for #ident {
            type Datum = #t;

            fn data(&self) -> &[#t] {
                unsafe {
                    &*self.data
                }
            }

            fn data_mut(&mut self) -> &mut [#t] {
                unsafe {
                    &mut *self.data
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
            
            fn iter(&self) -> TensorIterator<#t> {
                self.into_iter()
            }
            
            fn iter_mut(&mut self) -> TensorIterMut<#t> {
                self.into_iter()
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
                    self.data = std::slice::from_raw_parts_mut(#data_fn(self.tensor), 1) as *mut [#t];
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
                    let data = std::slice::from_raw_parts_mut(#data_fn(self.tensor), size) as *mut [#t];
                }
            }

            fn resize_2d(&mut self, size: [usize; 2]) {
                unsafe {
                    self.size = size.to_vec();
                    #resize_2d_fn(self.tensor, size[0] as i64, size[1] as i64);
                    let mut big_bound = 0;
                    let mut unit_size = 1;
                    let dim = #dim_fn(self.tensor) as usize;
                    self.stride = (0usize..dim).map(|i| {
                        let cur_stride = #stride_fn(self.tensor, i as i32) as usize;
                        let cur_bound = cur_stride * size[i];

                        if cur_bound > big_bound && i < dim - 1 {
                            big_bound = cur_bound;
                        }

                        if cur_stride == 1 {
                            unit_size = size[i];
                        }

                        cur_stride
                    }).collect();

                    let storage_bound = match big_bound {
                        0 => unit_size,
                        n => big_bound
                    };
                    let data = std::slice::from_raw_parts_mut(#data_fn(self.tensor), storage_bound) as *mut [#t];
                }
            }

            fn resize_3d(&mut self, size: [usize; 3]) {
                unsafe {
                    self.size = size.to_vec();
                    #resize_3d_fn(self.tensor, size[0] as i64, size[1] as i64, size[2] as i64);
                    
                    let mut big_bound = 0;
                    let mut unit_size = 1;
                    let dim = #dim_fn(self.tensor) as usize;
                    self.stride = (0usize..dim).map(|i| {
                        let cur_stride = #stride_fn(self.tensor, i as i32) as usize;
                        let cur_bound = cur_stride * size[i];

                        if cur_bound > big_bound && i < dim - 1 {
                            big_bound = cur_bound;
                        }

                        if cur_stride == 1 {
                            unit_size = size[i];
                        }

                        cur_stride
                    }).collect();

                    let storage_bound = match big_bound {
                        0 => unit_size,
                        n => big_bound
                    };
                    self.storage_bound = storage_bound;
                    let data = std::slice::from_raw_parts_mut(#data_fn(self.tensor), storage_bound) as *mut [#t];
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

            fn resize_nd(&mut self, size: &[usize], stride: &[usize]) {
                assert!((size.len() - 1 == stride.len()) || (size.len() == stride.len()), "Stride must have exactly {} elements or have exactly equals number of elements to size", size.len() - 1);
                
                unsafe {
                    self.size = size.to_owned();
                    self.stride = stride.to_owned();
                    if stride.len() < size.len() {
                        self.stride.push(1); // just to conform to Caffe2 API
                    }
                    #resize_nd_fn(self.tensor, size.len() as c_int, size.as_ptr() as *const i64, self.stride.as_ptr() as *const i64);

                    self.storage_bound = _data_bound(&self.size, &self.stride);
                    let data = std::slice::from_raw_parts_mut(#data_fn(self.tensor), self.storage_bound) as *mut [#t];
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

            fn storage(&mut self) -> &mut Option<#store_ty_id> {
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
        }

        #[cfg(feature = "serde")]
        impl<'de> Deserialize<'de> for #ident {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error> where D: Deserializer<'de> {
                struct TS;
                enum Fields {
                    Data,
                    Offset,
                    Size,
                    Stride
                };

                impl<'de> Deserialize<'de> for Fields {
                    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error> where D: Deserializer<'de> {
                        struct FieldsVisitor;

                        impl<'de> Visitor<'de> for FieldsVisitor {
                            type Value = Fields;

                            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                                write!(formatter, "Cannot deserialize field name")
                            }

                            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E> where E: de::Error {
                                match v {
                                    "data" => Ok(Fields::Data),
                                    "offset" => Ok(Fields::Offset),
                                    "size" => Ok(Fields::Size),
                                    "stride" => Ok(Fields::Stride),
                                    _ => Err(de::Error::unknown_field(v, FIELDS))
                                }
                            }
                        }

                        deserializer.deserialize_identifier(FieldsVisitor)
                    }
                }

                impl<'de> Visitor<'de> for TS {
                    type Value = #ident;

                    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                        write!(formatter, "Cannot deserialize {} struct", stringify!(#ident))
                    }

                    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error> where A: MapAccess<'de> {
                        let mut store: Option<#store_ty_id> = None;
                        let mut offset: Option<usize> = None;
                        let mut size: Option<Vec<usize>> = None;
                        let mut stride: Option<Vec<usize>> = None;

                        while let Some(name) = map.next_key()? {
                            match name {
                                Fields::Data => store = Some(map.next_value()?),
                                Fields::Offset => offset = Some(map.next_value()?),
                                Fields::Size => size = Some(map.next_value()?),
                                Fields::Stride => stride = Some(map.next_value()?)
                            }
                        }

                        let store = store.ok_or(de::Error::missing_field("data"))?;
                        let offset = offset.ok_or(de::Error::missing_field("offset"))?;
                        let size = size.ok_or(de::Error::missing_field("size"))?;
                        let stride = stride.ok_or(de::Error::missing_field("stride"))?;
                        
                        Ok(#ident::new_with_storage_nd(store, offset, &size, &stride))
                    }
                }

                const FIELDS: &'static [&str] = &["data", "offset", "size", "stride"];
                deserializer.deserialize_struct(stringify!(#ident), FIELDS, TS)
            }
        }

        #[cfg(feature = "serde")]
        impl Serialize for #ident {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> where S: Serializer {
                let mut state = serializer.serialize_struct(stringify!(#ident), 4)?;
                state.serialize_field("data", &self.storage)?;
                state.serialize_field("offset", &self.storage_offset())?;
                state.serialize_field("size", &self.size)?;
                state.serialize_field("stride", &self.stride)?;
                state.end()
            }
        }

        impl UtilityOp<#store_ty_id> for #ident {}

        impl ViewOp<#ident> for #ident {
            fn original(self) -> #ident {
                self
            }

            fn narrow(self, bound: &[Range<usize>]) -> Result<TensorView<#ident>, NarrowError> {
                let (cur_shape, cur_stride) = self.shape();
                let mut new_size = Vec::with_capacity(bound.len());
                let mut offset = self.storage_offset();

                // check every dimension whether the range is in boundary
                for (((dim, u_bound), cur_stride), new_bound) in cur_shape.iter().enumerate().zip(cur_stride.iter()).zip(bound.iter()) {
                    if *u_bound < new_bound.end {
                        return Err(NarrowError {dim: dim})
                    } else {
                        // calculate new size and new storage offset
                        new_size.push(new_bound.end - new_bound.start);
                        offset += new_bound.start * *cur_stride;
                    }
                }

                unsafe {
                    let storage = #store_ty_id::from(#storage_fn(self.tensor)).forget();
                    let tensor = #ident::new_with_storage_nd(storage, offset, &new_size, &cur_stride);

                    Ok(
                        TensorView {
                            original: self,
                            view: tensor
                        }
                    )
                }
            }

            unsafe fn unsafe_narrow(&self, bound: &[Range<usize>]) -> #ident {
                let (cur_shape, cur_stride) = self.shape();
                let mut new_size = Vec::with_capacity(bound.len());
                let mut offset = self.storage_offset();

                // check every dimension whether the range is in boundary
                for (((dim, u_bound), cur_stride), new_bound) in cur_shape.iter().enumerate().zip(cur_stride.iter()).zip(bound.iter()) {
                    if *u_bound < new_bound.end {
                        panic!("Size incompatible.")
                    } else {
                        // calculate new size and new storage offset
                        new_size.push(new_bound.end - new_bound.start);
                        offset += new_bound.start * *cur_stride;
                    }
                }

                let storage = #store_ty_id::from(#storage_fn(self.tensor)).forget();
                #ident::new_with_storage_nd(storage, offset, &new_size, &cur_stride)
            }

            fn narrow_on(self, dim: usize, new_bound: Range<usize>) -> Result<TensorView<#ident>, NarrowError> {
                let (cur_shape, _) = self.shape();
                // let mut new_size = cur_shape.to_owned();
                // let mut offset = self.storage_offset();

                if new_bound.end <= cur_shape[dim] {
                    // offset += new_bound.start * cur_stride[dim];
                    // new_size[dim] = new_bound.end;
                } else {
                    return Err(NarrowError {dim: dim })
                }

                unsafe {
                    // let storage = #store_ty_id::from(#storage_fn(self.tensor)).forget();
                    let tensor = #ident::new_narrow(&self, dim, new_bound.start, new_bound.end - new_bound.start);

                    Ok(
                        TensorView {
                            original: self,
                            view: tensor
                        }
                    )
                }
            }
            
            fn squeeze(self) -> TensorView<#ident> {
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

                    TensorView {
                        original: self,
                        view: new_ts
                    }
                }
            }
            
            unsafe fn unsafe_squeeze(&self) -> #ident {
                let mut new_ts = Self::new();
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

            unsafe fn unsafe_squeeze_dim(&self, dim: usize) -> #ident {
                let mut new_ts = Self::new();
                #squeeze_1d_fn(new_ts.tensor, self.tensor, dim as i64);
                new_ts.data = std::slice::from_raw_parts_mut(#data_fn(new_ts.tensor), self.storage_bound);
                new_ts.storage_bound = self.storage_bound;
                new_ts.stride = self.stride.to_owned();
                new_ts.size = self.size.to_owned();

                // only when size[dim] == 1 that squeeze will have effect
                if self.size[dim] == 1 {
                    new_ts.size.remove(dim);
                    new_ts.stride.remove(dim);
                }

                new_ts
            }

            fn view(self, sizes: &[Option<usize>]) -> Result<TensorView<#ident>, ViewError> {
                unsafe {
                    let new_size = self.infer_size(sizes)?;
                    let new_stride = self.compute_stride(&new_size)?;
                    let offset = self.storage_offset();
                    let mut storage = #store_ty_id::from(#storage_fn(self.tensor)).forget();
                    let ts = #ident::new_with_storage_nd(storage, offset, &new_size, &new_stride);
                    
                    Ok(
                        TensorView {
                            original: self,
                            view: ts
                        }
                    )
                }
            }
        }

        impl Tensor for #ident {
            type Datum = #t;
            type Storage = #store_ty_id;
        }

        impl<'a> Clone for #ident {
            fn clone(&self) -> #ident {
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
        impl Deref for #ident {
            type Target=[#t];

            fn deref(&self) -> &Self::Target {
                self.data()
            }
        }

        /// It deref into underlying storage data.
        /// It's similar to calling to [data](trait.Tensor.html#tymethod.data) function
        impl DerefMut for #ident {
            fn deref_mut(&mut self) -> &mut Self::Target {
                self.data_mut()
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

        impl Index<&[usize]> for #ident {
            type Output = #t;

            fn index(&self, idx: &[usize]) -> &Self::Output {
                let stride = self.shape().1;
                let actual_idx = idx.iter().enumerate().fold(0, |cum, (i, index)| {
                    cum + (index * stride[i])
                });
                &self.data()[actual_idx]
            }
        }

        impl IndexMut<usize> for #ident {
            fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
                &mut self.data_mut()[idx]
            }
        }

        impl IndexMut<&[usize]> for #ident {
            fn index_mut(&mut self, idx: &[usize]) -> &mut Self::Output {
                let stride = self.shape().1;
                let actual_idx = idx.iter().enumerate().fold(0, |cum, (i, index)| {
                    cum + (index * stride[i])
                });
                &mut self.data_mut()[actual_idx]
            }
        }

        impl Index<usize> for #ident {
            type Output = #t;

            fn index(&self, idx: usize) -> &Self::Output {
                &self.data()[idx]
            }
        }

        impl<'a> IntoIterator for &'a #ident {
            type Item = #t;
            type IntoIter = TensorIterator<'a, #t>;

            fn into_iter(self) -> Self::IntoIter {
                let (size, stride) = self.shape();
                TensorIterator::new(self.data(), size, stride)
            }
        }

        impl<'a> IntoIterator for &'a mut #ident {
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

        impl<'a> From<&'a [#t]> for #ident {
            fn from(slice: &'a [#t]) -> #ident {
                let mut tensor = #ident::new_with_size_1d(slice.len());
                tensor.data_mut().iter_mut().enumerate().for_each(|(i, v)| *v = slice[i]);
                tensor
            }
        }
    };

    TokenStream::from(expanded)
}