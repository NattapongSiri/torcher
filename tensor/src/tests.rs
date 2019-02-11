use super::*;

#[test]
fn byte_to_char() {
    let mut bytes = ByteTensor::new_with_size_3d([4, 2, 1]);
    bytes.data_mut().iter_mut().enumerate().for_each(|(i, b)| *b = i as u8);
    let chars = CharTensor::from(&bytes);
    // elementwise, it's equals.
    chars.iter().zip(bytes.iter()).for_each(|(c, b)| assert_eq!(c, b as i8));
}

#[test]
fn float_create_drop() {
    FloatTensor::new();
}

#[test]
fn float_create_1d_drop() {
    let mut storage = FloatStorage::new_with_size(10);
    storage.fill(1.0);
    FloatTensor::new_with_storage_1d(storage, 2, 8);
}

#[test]
fn float_create_1d_desc() {
    let storage = FloatStorage::new_with_size(10);
    let ts = FloatTensor::new_with_storage_1d(storage, 2, 8);
    assert_eq!("torch.xTensor of size 8", ts.desc());
}

#[test]
fn float_create_2d_desc() {
    let storage = FloatStorage::new_with_size(10);
    let ts = FloatTensor::new_with_storage_2d(storage, 2, [4, 2], 2);
    assert_eq!("torch.xTensor of size 4x2", ts.desc());
}

#[test]
fn float_create_3d_desc() {
    let storage = FloatStorage::new_with_size(10);
    let ts = FloatTensor::new_with_storage_3d(storage, 1, [3, 3, 1], [3, 1]);
    assert_eq!("torch.xTensor of size 3x3x1", ts.desc());
}

#[test]
fn float_create_4d_desc() {
    let storage = FloatStorage::new_with_size(10);
    let ts = FloatTensor::new_with_storage_4d(storage, 1, [2, 2, 2, 1], [4, 2, 1]);
    assert_eq!("torch.xTensor of size 2x2x2x1", ts.desc());
}

#[test]
fn float_create_empty_1d_desc() {
    let ts = FloatTensor::new_with_size_1d(8);
    assert_eq!("torch.xTensor of size 8", ts.desc());
}

#[test]
fn float_create_empty_2d_desc() {
    let ts = FloatTensor::new_with_size_2d([4, 2]);
    assert_eq!("torch.xTensor of size 4x2", ts.desc());
}

#[test]
fn float_create_empty_3d_desc() {
    let ts = FloatTensor::new_with_size_3d([3, 2, 1]);
    assert_eq!("torch.xTensor of size 3x2x1", ts.desc());
}

#[test]
fn float_create_empty_4d_desc() {
    let ts = FloatTensor::new_with_size_4d([4, 3, 2, 1]);
    assert_eq!("torch.xTensor of size 4x3x2x1", ts.desc());
}

#[test]
fn float_create_unfold() {
    let mut ts = FloatTensor::new_with_size_3d([5, 2, 1]);
    unsafe {
        let mut uf_1 = ts.new_unfold(0, 2, 2);
        ts[0] = 2f32;
        uf_1[0] = 2f32;
        assert_eq!(ts[0], uf_1[0]);
        assert_eq!(&[2usize, 2, 1, 2], uf_1.shape().0);
    }
}

#[test]
fn float_data() {
    let mut storage = FloatStorage::new_with_size(10);
    storage.iter_mut().enumerate().for_each(|(i, v)| *v = i as f32);
    let tensor = FloatTensor::new_with_storage_3d(storage, 1, [3, 2, 2], [2, 1]);
    
    let validator = &[1.0f32, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0];
    
    (&tensor).into_iter().enumerate().for_each(|(i, v)| assert_eq!(validator[i], v));
}

#[test]
fn float_data_4d_desc() {
    let mut ts = FloatTensor::new_with_size_4d([4, 3, 2, 1]);
    let raw_data = ts.data_mut();

    for i in 0..raw_data.len() {
        raw_data[i] = i as f32;
    }

    let validator : Vec<f32> = (0..24).map(|i| i as f32).collect();

    assert_eq!(validator, raw_data);
}

#[test]
fn float_clone() {
    let mut ts = FloatTensor::new_with_size_4d([4, 3, 2, 1]);
    let raw_data = ts.data_mut();

    for i in 0..raw_data.len() {
        raw_data[i] = i as f32;
    }

    let cloned = ts.clone();

    let validator : Vec<f32> = (0..24).map(|i| i as f32).collect();

    assert_eq!(validator, cloned.data());
}

#[test]
fn float_contiguous() {
    let storage = FloatStorage::new_with_size(10);
    let mut ts = FloatTensor::new_with_storage_4d(storage, 1, [2, 2, 2, 1], [4, 2, 1]);
    ts[1] = 0f32;
    let mut cont = ts.new_contiguous();
    cont[0] = 1f32;

    assert_ne!(cont[0], ts[1]); // check whether the storage is shared
    assert_eq!(&[4usize, 2, 1, 1] as &[usize] , cont.stride.as_slice());
}

#[test]
fn float_deref() {
    let mut storage = FloatStorage::new_with_size(10);
    storage.iter_mut().enumerate().for_each(|(i, v)| *v = i as f32);
    let ts = FloatTensor::new_with_storage_4d(storage, 1, [2, 2, 2, 1], [4, 2, 1]);
    let validator = &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    
    for i in 0..ts.len() {
        assert_eq!(validator[i], ts[i]);
    }
}

#[test]
fn float_deref_mut() {
    let mut ts = FloatTensor::new_with_size_4d([2, 2, 2, 1]);
    for i in 0..ts.len() {
        ts[i] = i as f32;
    }
    let validator = &[0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    
    for i in 0..ts.len() {
        assert_eq!(validator[i], ts[i]);
    }
}

#[test]
fn float_get_0d() {
    let mut storage = FloatStorage::new_with_size(10);
    for i in 0..storage.len() {
        storage[i] = 1 as f32;
    }
    let mut ts = FloatTensor::new_with_storage_4d(storage, 1, [2, 2, 2, 1], [2, 4, 1]);

    ts.resize_0d();

    assert_eq!(1f32, ts.get_0d());
}

#[test]
fn float_get_1d() {
    let mut storage = FloatStorage::new_with_size(10);
    for i in 0..storage.len() {
        storage[i] = i as f32;
    }
    let mut ts = FloatTensor::new_with_storage_4d(storage, 1, [2, 2, 2, 1], [1, 4, 1]);

    ts.resize_1d(8); // [[[[1], [2]], [[5], [6]]], [[[3], [4]], [[7], [8]]]]
    let validate = [1f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    for i in 0..ts.shape().0[0] {
        assert_eq!(validate[i], ts.get_1d(i));
    }
}

#[test]
fn float_get_2d() {
    let mut storage = FloatStorage::new_with_size(10);
    for i in 0..storage.len() {
        storage[i] = i as f32;
    }
    let mut ts = FloatTensor::new_with_storage_4d(storage, 1, [2, 2, 2, 1], [1, 2, 1]);
    let (m, n) = (4, 2);
    ts.resize_2d([m, n]);
    let validate = [1f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    for i in 0..m {
        for j in 0..n {
            assert_eq!(validate[i * n + j], ts.get_2d([i, j]));
        }
    }
}

#[test]
fn float_get_3d() {
    let mut storage = FloatStorage::new_with_size(10);
    for i in 0..storage.len() {
        storage[i] = i as f32;
    }
    let mut ts = FloatTensor::new_with_storage_4d(storage, 1, [2, 2, 2, 1], [1, 2, 1]);
    let (m, n, o) = (2, 2, 2);
    ts.resize_3d([m, n, o]);
    let validate = [1f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    for i in 0..m {
        for j in 0..n {
            for k in 0..o {
                assert_eq!(validate[i * n * o + j * o + k], ts.get_3d([i, j, k]));
            }
        }
    }
}

#[test]
fn float_get_4d() {
    let mut storage = FloatStorage::new_with_size(10);
    for i in 0..storage.len() {
        storage[i] = i as f32;
    }
    let ts = FloatTensor::new_with_storage_4d(storage, 1, [2, 2, 2, 1], [4, 1, 2]);
    let (m, n, o, p) = (2, 2, 2, 1);
    let validate = [1f32, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0];
    for i in 0..m {
        for j in 0..n {
            for k in 0..o {
                for l in 0..p {
                    assert_eq!(validate[i * n * o * p + j * o * p + k * p + l], ts.get_4d([i, j, k, l]));
                }
            }
        }
    }
}

#[test]
fn float_select() {
    let mut storage = FloatStorage::new_with_size(10);
    storage[3] = 1f32;
    unsafe {
        let ts = FloatTensor::new_with_storage_2d(storage, 2, [4, 2], 2);
        let sel = ts.new_select(1, 1);

        assert_eq!(1f32 , sel[0]);
    }
}

#[test]
fn float_narrow() {
    let mut storage = FloatStorage::new_with_size(10);
    storage.iter_mut().enumerate().for_each(|(i, v)| *v = i as f32);

    let ts = FloatTensor::new_with_storage_2d(storage, 1, [4, 2], 2);
    let tv = ts.narrow(&[1..4, 0..2]).unwrap();
    
    assert_eq!("torch.xTensor of size 3x2", tv.desc());
    let expected = &[3f32, 4.0, 5.0, 6.0, 7.0, 8.0];
    tv.iter().zip(expected.iter()).for_each(|(v, e)| assert_eq!(v, *e));
}

#[test]
fn float_unsafe_narrow() {
    let mut storage = FloatStorage::new_with_size(10);
    storage.iter_mut().enumerate().for_each(|(i, v)| *v = i as f32);

    let ts = FloatTensor::new_with_storage_2d(storage, 1, [4, 2], 2);
    unsafe {
        let tv = ts.unsafe_narrow(&[1..4, 0..2]);
        
        assert_eq!("torch.xTensor of size 3x2", tv.desc());
        let expected = &[3f32, 4.0, 5.0, 6.0, 7.0, 8.0];
        tv.iter().zip(expected.iter()).for_each(|(v, e)| assert_eq!(v, *e));
    }
}

#[test]
fn float_narrow_on() {
    let mut storage = FloatStorage::new_with_size(10);
    storage.iter_mut().enumerate().for_each(|(i, v)| *v = i as f32);

    // shape will be [[1, 2], [3, 4], [5, 6], [7, 8]]
    let ts = FloatTensor::new_with_storage_2d(storage, 1, [4, 2], 2);
    // shape will be [[3, 4], [5, 6]]
    let tv = ts.narrow_on(0, 1..3).unwrap()
    // shape will be [[4], [6]]
               .narrow_on(1, 1..2).unwrap();
    assert_eq!(tv.shape().0, &[2, 1]);
    let expected = [4f32, 6.0];
    tv.iter().enumerate().for_each(|(i, v)| {
        assert_eq!(v, expected[i]);
    });
    assert_eq!(tv.squeeze().shape().0, &[2usize]);
}

#[test]
fn float_new_narrow() {
    let mut storage = FloatStorage::new_with_size(10);
    storage[4] = 2f32;
    storage[5] = 1f32;
    unsafe {
        let ts = FloatTensor::new_with_storage_2d(storage, 1, [4, 2], 2);
        let sel = ts.new_narrow(1, 1, 1).new_narrow(0, 1, 2);
        
        assert_eq!("torch.xTensor of size 2x1", sel.desc());
        assert_eq!([2f32, 1f32] , sel[0..2]);
    }
}

#[test]
fn float_transpose() {
    let mut storage = FloatStorage::new_with_size(10);
    for i in 0..storage.len() {
        storage[i] = i as f32;
    }
    unsafe {
        let ts = FloatTensor::new_with_storage_2d(storage, 2, [4, 2], 2);
        let tp = ts.new_transpose(0, 1);
        
        assert_eq!("torch.xTensor of size 2x4", tp.desc());
    }
}

#[test]
fn float_deep_view() {
    let mut ts = FloatTensor::new_with_size_1d(36);
    ts.iter_mut().enumerate().for_each(|(i, v)| *v = i as f32);
    let ts_v = ts.view(&[Some(2), Some(18)]).unwrap();
    // view on view
    let ts_v2 = ts_v.view(&[Some(2), Some(3), Some(3), None],).unwrap();
    // view on view on view.
    let ts_v3 = ts_v2.view(&[Some(2), Some(2), Some(3), None]).unwrap();
    let expected_size : &[usize] = &[2, 2, 3, 3];
    let expected_stride: &[usize] = &[18, 9, 3, 1];
    assert_eq!((expected_size, expected_stride), ts_v3.shape());
    // elements shall still in ascending order
    ts_v3.iter().enumerate().for_each(|(i, v)| assert_eq!(i as f32, v)); 
}

#[test]
fn float_dim() {
    let storage = FloatStorage::new_with_size(10);
    let ts = FloatTensor::new_with_storage_4d(storage, 1, [2, 2, 2, 1], [4, 2, 1]);
    assert_eq!(4, ts.dimensions());
}

#[test]
fn float_is_contiguous() {
    let storage = FloatStorage::new_with_size(10);
    let ts = FloatTensor::new_with_storage_4d(storage, 1, [2, 2, 2, 1], [4, 2, 1]);
    assert!(ts.is_contiguous());
}

#[test]
fn float_iterator() {
    let mut storage = FloatStorage::new_with_size(10);
    for i in 0..storage.len() {
        storage[i] = i as f32;
    }
    let ts = FloatTensor::new_with_storage_4d(storage, 1, [2, 2, 2, 1], [4, 2, 1]);
    let validator = &[1f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    (&ts).into_iter().enumerate().for_each(|(i, v)| {
        assert_eq!(validator[i], v);
    });
}

#[test]
fn float_iter_mut() {
    let mut storage = FloatStorage::new_with_size(10);
    for i in 0..storage.len() {
        storage[i] = i as f32;
    }
    let mut ts = FloatTensor::new_with_storage_4d(storage, 1, [2, 2, 2, 1], [4, 2, 1]);
    let validator = &[1f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    (&mut ts).into_iter().enumerate().for_each(|(i, v)| {
        assert_eq!(validator[i], *v);
    });
}

#[test]
fn float_resize_0d() {
    let mut storage = FloatStorage::new_with_size(10);
    for i in 0..storage.len() {
        storage[i] = i as f32;
    }
    let mut ts = FloatTensor::new_with_storage_4d(storage, 1, [2, 2, 2, 1], [4, 2, 1]);
    ts.resize_0d();

    assert_eq!(0, ts.dimensions());
    assert_eq!(1f32, ts[0])
}

#[test]
fn float_resize_1d() {
    let mut storage = FloatStorage::new_with_size(10);
    for i in 0..storage.len() {
        storage[i] = i as f32;
    }
    let mut ts = FloatTensor::new_with_storage_4d(storage, 1, [2, 2, 2, 1], [4, 2, 1]);
    ts.resize_1d(2);

    assert_eq!(1, ts.dimensions());
    assert_eq!(1f32, ts[0])
}

#[test]
fn float_resize_2d() {
    let mut storage = FloatStorage::new_with_size(10);
    for i in 0..storage.len() {
        storage[i] = i as f32;
    }
    let mut ts = FloatTensor::new_with_storage_4d(storage, 1, [2, 2, 2, 1], [4, 2, 1]);
    ts.resize_2d([4, 2]);

    assert_eq!(2, ts.dimensions());
    assert_eq!((&[4usize, 2] as &[usize], &[2usize, 1] as &[usize]), ts.shape());
    assert_eq!(1f32, ts[0])
}

#[test]
fn float_resize_3d() {
    let mut storage = FloatStorage::new_with_size(10);
    for i in 0..storage.len() {
        storage[i] = i as f32;
    }
    let mut ts = FloatTensor::new_with_storage_4d(storage, 1, [2, 2, 2, 1], [4, 2, 1]);
    ts.resize_3d([4, 1, 2]);

    assert_eq!(3, ts.dimensions());
    assert_eq!((&[4usize, 1, 2] as &[usize], &[2usize, 2, 1] as &[usize]), ts.shape());
    assert_eq!(1f32, ts[0])
}

// #[test]
// fn float_resize_4d() {
//     let mut storage = FloatStorage::new_with_size(10);
//     for i in 0..storage.len() {
//         storage[i] = i as f32;
//     }
//     let mut ts = FloatTensor::new_with_storage_3d(storage, 1, [4, 2, 1], [2, 1, 1]);

//     ts.resize_4d([2, 2, 2, 1]);

//     assert_eq!(4, ts.dimensions());
//     assert_eq!((&[2usize, 2, 2, 1] as &[usize], &[4usize, 2, 1, 1] as &[usize]), ts.shape());
//     assert_eq!(1f32, ts.data()[0])
// }

// #[test]
// fn float_resize_5d() {
//     let mut storage = FloatStorage::new_with_size(10);
//     let mut ts = FloatTensor::new_with_size_2d([8, 1]);
//     let mut data = ts.data();
//     for i in 0..data.len() {
//         data[i] = i as f32;
//     }
//     ts.resize_5d([2, 1, 2, 1, 2]);

//     assert_eq!(5, ts.dimensions());
//     assert_eq!((&[2usize, 1, 2, 1, 2] as &[usize], &[4usize, 4, 2, 2, 1] as &[usize]), ts.shape());
//     assert_eq!(1f32, ts.data()[0])
// }

#[test]
fn float_resize_nd() {
    let mut storage = FloatStorage::new_with_size(10);
    for i in 0..storage.len() {
        storage[i] = i as f32;
    }
    let mut ts = FloatTensor::new_with_storage_4d(storage, 1, [2, 2, 2, 1], [4, 2, 1]);
    ts.resize_nd(&[4, 2], &[2]);
    let size = &[4, 2];

    for i in 0..ts.dimensions() {
        assert_eq!(ts.size(i), size[i]);
    }
}

#[cfg(feature = "serde")]
#[test]
fn float_serialize() {
    let mut tensor = ByteTensor::new_with_size_3d([2, 3, 4]);
    tensor.iter_mut().for_each(|d| *d = 1);
    assert_eq!(serde_yaml::to_string(&tensor).unwrap(),
    "---\ndata:\n  - 1\n  - 1\n  - 1\n  - 1\n  - 1\n  - 1\n  - 1\n  - 1\n  - 1\n  - 1\n  - 1\n  - 1\n  - 1\n  - 1\n  - 1\n  - 1\n  - 1\n  - 1\n  - 1\n  - 1\n  - 1\n  - 1\n  - 1\n  - 1\noffset: 0\nsize:\n  - 2\n  - 3\n  - 4\nstride:\n  - 12\n  - 4\n  - 1");
}

#[cfg(feature = "serde")]
#[test]
fn float_derialize() {
    let mut tensor = ByteTensor::new_with_size_3d([2, 3, 4]);
    tensor.iter_mut().for_each(|d| *d = 1);
    let serialized = serde_yaml::to_string(&tensor).unwrap();
    let loaded: ByteTensor = serde_yaml::from_str(&serialized).unwrap();
    
    loaded.iter().for_each(|v| assert_eq!(v, 1));
}

#[test]
fn float_set_0d() {
    let mut storage = FloatStorage::new_with_size(10);
    for i in 0..storage.len() {
        storage[i] = i as f32;
    }
    let mut ts = FloatTensor::new_with_storage_4d(storage, 1, [2, 2, 2, 1], [1, 4, 2]);

    ts.resize_0d(); // [[[[1], [2]], [[5], [6]]], [[[3], [4]], [[7], [8]]]]
    ts.set_0d(2f32);
    assert_eq!(2f32, ts[0])
}

#[test]
fn float_set_get_1d() {
    let storage = FloatStorage::new_with_size(10);
    let mut ts = FloatTensor::new_with_storage_1d(storage, 1, 8);

    for i in 0..ts.shape().0[0] {
        ts.set_1d(i as usize, i as f32);
    }

    let validate = [0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    for i in 0..ts.shape().0[0] {
        assert_eq!(validate[i], ts.get_1d(i));
    }
}

#[test]
fn float_set_2d() {
    let storage = FloatStorage::new_with_size(10);
    let (m, n) = (4, 2);
    let mut ts = FloatTensor::new_with_storage_2d(storage, 1, [m, n], 2);
    let validate = [1f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    for i in 0..m {
        for j in 0..n {
            ts.set_2d([i, j], validate[i * n + j]);
        }
    }

    assert_eq!(validate, ts[0..8])
}

#[test]
fn float_set_3d() {
    let storage = FloatStorage::new_with_size(10);
    let (m, n, o) = (2, 2, 2);
    let mut ts = FloatTensor::new_with_storage_3d(storage, 1, [m, n, o], [4, 2]);
    let validate = [1f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    for i in 0..m {
        for j in 0..n {
            for k in 0..o {
                ts.set_3d([i, j, k], validate[i * n * o + j * o + k]);
            }
        }
    }

    assert_eq!(validate, ts.data()[0..8])
}

#[test]
fn float_set_4d() {
    let storage = FloatStorage::new_with_size(10);
    let (m, n, o, p) = (2, 2, 2, 1);
    let mut ts = FloatTensor::new_with_storage_4d(storage, 1, [m, n, o, p], [4, 1, 2]);
    let src = [1f32, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0];
    let validate = [1f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    
    for i in 0..m {
        for j in 0..n {
            for k in 0..o {
                for l in 0..p {
                    ts.set_4d([i, j, k, l], src[i * n * o * p + j * o * p + k * p + l]);
                }
            }
        }
    }
    assert_eq!(validate, ts[0..8])
}

#[test]
fn float_size() {
    let storage = FloatStorage::new_with_size(10);
    let ts = FloatTensor::new_with_storage_4d(storage, 1, [2, 2, 2, 1], [4, 2, 1]);
    let size = &[2, 2, 2, 1];

    for i in 0..ts.dimensions() {
        assert_eq!(ts.size(i), size[i]);
    }
}

#[test]
fn float_stride() {
    let storage = FloatStorage::new_with_size(10);
    let ts = FloatTensor::new_with_storage_4d(storage, 1, [2, 2, 2, 1], [4, 2, 1]);
    let stride = &[4, 2, 1, 1];

    for i in 0..ts.dimensions() {
        assert_eq!(ts.stride(i), stride[i]);
    }
}

#[test]
fn float_squeeze() {
    let mut storage = FloatStorage::new_with_size(10);
    for i in 0..storage.len() {
        storage[i] = i as f32;
    }
    let ts = FloatTensor::new_with_storage_4d(storage, 2, [2, 1, 2, 1], [4, 2, 1]);

    let squeezed = ts.squeeze();

    assert_eq!(&[2usize, 2], squeezed.shape().0);
}

#[test]
fn float_view() {
    let mut ts = FloatTensor::new_with_size_1d(10);
    ts.iter_mut().enumerate().for_each(|(i, v)| *v = i as f32);
    let ts_v = ts.view(&[Some(5), Some(2)]).unwrap();
    let expected_size : &[usize] = &[5, 2];
    let expected_stride: &[usize] = &[2, 1];
    assert_eq!((expected_size, expected_stride), ts_v.shape());
}

#[test]
#[ignore]
fn bench_deref() {
    use std::time::Instant;
    let timer = Instant::now();
    let mut tensor = FloatTensor::new_with_size_1d(4e8 as usize);
    for i in 0..tensor.len() {
        tensor[i] = i as f32;
    }
    let duration = timer.elapsed();
    eprintln!("Done mut tensor[i]=value in {}.{}s", duration.as_secs(), duration.subsec_millis());
    let timer = Instant::now();
    let mut tensor = FloatTensor::new_with_size_1d(4e8 as usize);
    let data : &mut [f32] = &mut tensor;
    for i in 0..data.len() {
        data[i] = i as f32;
    }
    let duration = timer.elapsed();
    eprintln!("Done mut data[i]=value in {}.{}s", duration.as_secs(), duration.subsec_millis());

    let raw : Vec<f32> = (0..(4e8 as usize)).map(|f| f as f32).collect();
    
    let timer = Instant::now();
    let _tensor = FloatTensor::from(raw.as_slice());
    let duration = timer.elapsed();
    eprintln!("Done implict cast using From trait in {}.{}s", duration.as_secs(), duration.subsec_millis());
}