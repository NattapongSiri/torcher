use super::*;

#[test]
fn float_construct_destroy() {
    FloatStorage::new();
}

#[test]
fn double_construct_destroy() {
    DoubleStorage::new();
}

#[test]
fn double_from() {
    let fs = FloatStorage::new_with_size(5);
    unsafe {
        // auto convert using From trait
        println!("Attempt to convert from native c store");
        let native_store = fs.storage();
        let another_fs = FloatStorage::from(native_store);
        another_fs.forget(); // or it will crash because underlying storage got free twice
        println!("Conversion succeed.");
    }
}

#[test]
fn float_fill_display() {
    let mut fs = FloatStorage::new_with_size(5);
    fs.fill(1f32);
    assert_eq!("FloatStorage:size=5:data=[1.0, 1.0, 1.0, 1.0, 1.0]", fs.to_string());
}

#[test]
fn double_fill_display() {
    let mut fs = DoubleStorage::new_with_size(5);
    fs.fill(1.0f64);
    assert_eq!("DoubleStorage:size=5:data=[1.0, 1.0, 1.0, 1.0, 1.0]", fs.to_string());
}

#[test]
fn double_fill_display_large() {
    let mut fs = DoubleStorage::new_with_size(50);
    fs.fill(2.0f64);
    assert_eq!("DoubleStorage:size=50:first(10)=[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]:last(10)=[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]", fs.to_string());
}

#[test]
fn float_mutate_data() {
    let mut fs = FloatStorage::new_with_size(32);
    let mut validator = Vec::with_capacity(32);

    fs.data_mut().iter_mut().enumerate().for_each(|(i, d)| {
        *d = i as f32;
        validator.push(i as f32);
    });

    assert_eq!(fs.data(), validator.as_slice());
}

#[test]
fn double_mutate_data() {
    let mut fs = DoubleStorage::new_with_size(32);
    let mut validator = Vec::with_capacity(32);

    fs.data_mut().iter_mut().enumerate().for_each(|(i, d)| {
        *d = i as f64;
        validator.push(i as f64);
    });

    assert_eq!(fs.data(), validator.as_slice());
}

#[test]
fn float_resize_mutate_data() {
    let mut fs = FloatStorage::new_with_size(16);
    fs.resize(32);
    let mut validator = Vec::with_capacity(32);

    fs.data_mut().iter_mut().enumerate().for_each(|(i, d)| {
        *d = i as f32;
        validator.push(i as f32);
    });

    assert_eq!(fs.data(), validator.as_slice());
}

#[test]
fn double_resize_mutate_data() {
    let mut fs = DoubleStorage::new_with_size(16);
    fs.resize(32);
    let mut validator = Vec::with_capacity(32);

    fs.data_mut().iter_mut().enumerate().for_each(|(i, d)| {
        *d = i as f64;
        validator.push(i as f64);
    });

    assert_eq!(fs.data(), validator.as_slice());
}

#[test]
fn float_mutate_resize_data() {
    let mut fs = FloatStorage::new_with_size(32);
    let mut validator = Vec::with_capacity(32);

    fs.data_mut().iter_mut().enumerate().for_each(|(i, d)| {
        *d = i as f32;
        validator.push(i as f32);
    });

    fs.resize(16);

    assert_eq!(fs.data(), &validator[0..16]);
}

#[test]
fn double_mutate_resize_data() {
    let mut fs = DoubleStorage::new_with_size(32);
    let mut validator = Vec::with_capacity(32);

    fs.data_mut().iter_mut().enumerate().for_each(|(i, d)| {
        *d = i as f64;
        validator.push(i as f64);
    });

    fs.resize(16);

    assert_eq!(fs.data(), &validator[0..16]);
}

#[test]
fn float_index() {
    let mut fs = FloatStorage::new_with_size(32);
    let mut validator = Vec::with_capacity(32);

    fs.data_mut().iter_mut().enumerate().for_each(|(i, d)| {
        *d = i as f32;
        validator.push(i as f32);
    });

    assert_eq!(31f32, fs[fs.len() - 1]);
}

#[test]
fn double_index() {
    let mut ds = DoubleStorage::new_with_size(32);
    let mut validator = Vec::with_capacity(32);

    ds.data_mut().iter_mut().enumerate().for_each(|(i, d)| {
        *d = i as f64;
        validator.push(i as f64);
    });

    assert_eq!(31f64, ds[ds.len() - 1]);
}

#[test]
fn float_index_mut() {
    let mut fs = FloatStorage::new_with_size(32);

    for i in 0..fs.len() {
        fs[i] = i as f32;
    }

    assert_eq!(31f32, fs[fs.len() - 1]);
}

#[test]
fn double_index_mut() {
    let mut ds = DoubleStorage::new_with_size(32);

    for i in 0..ds.len() {
        ds[i] = i as f64;
    }

    assert_eq!(31f64, ds[ds.len() - 1]);
}

#[test]
#[ignore]
fn bench_index() {
    let mut ds = DoubleStorage::new_with_size(1e8 as usize);

    use std::time::Instant;
    let begin = Instant::now();
    let data = ds.data_mut();

    for i in 0..data.len() {
        data[i] = 2f64;
    }

    let end = begin.elapsed();
    println!("Done Mut Slice in {}.{}s", end.as_secs(), end.subsec_millis());
    
    let begin = Instant::now();

    for i in 0..ds.len() {
        ds[i] = 1f64;
    }

    let end = begin.elapsed();
    println!("Done IndexMut in {}.{}s", end.as_secs(), end.subsec_millis());
}

#[test]
#[ignore]
fn memory_leak_test() {
    println!("
    This test never fail.
    The only way to check is monitor memory consumption.
    If memory leak, it'll consume about 34GB of memory.
    This shall be large enough to freeze most of PC.
    Otherwise, it'll consume about 1.7GB.
    ");
    // if memory leak, it'll consume about 34GB of memory
    let n = 20;
    for i in 0..n {
        // each new consume about 8 bytes per * size + 1 bool + 1 usize which roughly equals 17 bytes in 64bits system.
        // 63e6 size is estimated to be 1.7GB per each instance
        let mut tmp = DoubleStorage::new_with_size(1e8 as usize);
        // if we didn't fill the value, it may not actually allocate memory
        println!("Mem leak test {}/{}", i, n);
        tmp.fill(1.0);
    }

    println!("Mem leak test done");
}