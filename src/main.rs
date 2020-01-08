use opencv::{prelude::*, core, highgui, imgcodecs};
use std::mem::transmute;
use tch::{Device, Kind};

fn saturate<T: num_traits::float::Float>(val: T, min: T, max: T) -> T {
    val.max(min).min(max)
}

fn map_range<T: num_traits::float::Float>(val: T, in_a: T, in_b: T, out_a: T, out_b: T) -> T {
    let pos = (val - in_a) / (in_b - in_a);
    saturate(out_a + (out_b - out_a) * pos, out_a, out_b)
}

fn main() {
    println!("Start Depth Prediction!");

    let window = "Depth Prediction";
    highgui::named_window(window, 1).unwrap();

    let encoder_model = tch::CModule::load("models/encoder.pt").unwrap();
    let depth_model = tch::CModule::load("models/depth.pt").unwrap();

    let w = 640;
    let h = 192;

    let img = tch::vision::image::load("assets/test_image_01.png")
        .unwrap()
        .to_device(Device::Cuda(0));
    let img = img.to_kind(Kind::Float) / 255;
    let i_img: tch::IValue = tch::IValue::Tensor(img.unsqueeze(0));
    let encoder_output = encoder_model.forward_is(&[i_img]).unwrap();
    let enc_tensors = match &encoder_output {
        tch::IValue::Tuple(enc_tensors) => Some(enc_tensors),
        _ => None,
    }
    .unwrap();
    println!("encoder_output: {:?}", encoder_output);

    if let tch::IValue::Tensor(tensor) = &enc_tensors[0] {
        println!("encoder_min: {:?}", tensor.min());
        println!("encoder_max: {:?}", tensor.max());
    }

    let depth_outputs = depth_model
        .forward_is(&[
            &enc_tensors[0],
            &enc_tensors[1],
            &enc_tensors[2],
            &enc_tensors[3],
            &enc_tensors[4],
        ])
        .unwrap();
    println!("depth_outputs: {:?}", depth_outputs);

    let mut depth_output = None;
    if let tch::IValue::Tuple(tensors) = &depth_outputs {
        if let tch::IValue::Tensor(tensor) = &tensors[0] {
            depth_output = Some(tensor);
        }
    };
    let depth_output = depth_output.unwrap();
    let depth_min = depth_output.min();
    let depth_max = depth_output.max();
    println!("depth_min: {:?}", depth_min);
    println!("depth_max: {:?}", depth_max);
    let depth_output = depth_output.squeeze().to_device(Device::Cpu);
    println!("depth_output: {:?}", depth_output);

    let mut depth_ptr : Vec<f32> = depth_output.into();
    let depth_mat = core::Mat::new_rows_cols_with_data(
        h,
        w,
        core::CV_32FC1,
        unsafe { transmute(depth_ptr.as_mut_ptr()) },
        0,
    )
    .unwrap();

    let depth_min = f32::from(depth_min);
    let depth_max = f32::from(depth_max);

    let magma = imgcodecs::imread("assets/magma.png", imgcodecs::IMREAD_COLOR).unwrap();

    let mut depth_img =
        opencv::core::Mat::new_rows_cols_with_default(h, w, core::CV_8UC3, core::Scalar::default())
            .unwrap();
    for x in 0..w {
        for y in 0..h {
            let d: f32 = *depth_mat.at_2d(y, x).unwrap();
            let d = map_range(d, depth_min, depth_max, 0.0f32, 1.0f32);
            let m = 0f32.max(727f32.min(d * 727f32)) as i32;
            let m = magma.at_2d::<core::Vec3b>(0, m).unwrap();
            let w = depth_img.at_2d_mut::<core::Vec3b>(y, x).unwrap();
            *w = *m;
        }
    }

    highgui::imshow(window, &depth_img).unwrap();

    let params = opencv::types::VectorOfint::new();
    imgcodecs::imwrite("depth.jpg", &depth_img, &params).unwrap();

    highgui::wait_key(10000).unwrap();
}
