use failure::Fallible;
use opencv::{core, highgui, imgcodecs, imgproc, prelude::*};
use std::mem::transmute;
use tch::{Tensor, Device, Kind};

fn tensor_map_range(
    val: &Tensor,
    in_a: &Tensor,
    in_b: &Tensor,
    out_a: &Tensor,
    out_b: &Tensor,
) -> Fallible<Tensor> {
    let pos = val.f_sub(in_a)?.f_div(&in_b.f_sub(in_a)?)?;
    let mapped = out_a.f_add(&out_b.f_sub(out_a)?.f_mul(&pos)?)?;
    Ok(mapped.clamp(f64::from(out_a), f64::from(out_b)))
}

fn main() {
    println!("Start Depth Prediction!");

    let window = "Depth Prediction";
    highgui::named_window(window, 1).unwrap();

    let encoder_model = tch::CModule::load("models/encoder.pt").unwrap();
    let depth_model = tch::CModule::load("models/depth.pt").unwrap();

    let w = 640;
    let h = 192;

    let color_map = tch::vision::image::load("assets/magma.png")
        .unwrap()
        .to_device(Device::Cuda(0));
    println!("color_map: {:?}", color_map);

    let img = tch::vision::image::load("assets/test_image_01.png")
        .unwrap()
        .to_device(Device::Cuda(0));
    println!("img: {:?}", img);
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
    let depth_map_min = Tensor::from(0f64).to_device(tch::Device::Cuda(0));
    let depth_map_max = Tensor::from(1f64).to_device(tch::Device::Cuda(0));
    let depth_output = tensor_map_range(
        depth_output,
        &depth_min,
        &depth_max,
        &depth_map_min,
        &depth_map_max,
    )
    .unwrap();

    let color_index = depth_output
        .f_mul(&Tensor::from(727f32))
        .unwrap()
        .flatten(0, 3)
        .to_kind(tch::Kind::Int64);

    let depth_color = color_map
        .index_select(2, &color_index)
        .permute(&[2, 1, 0])
        .to_device(tch::Device::Cpu);

    let depth_color = Vec::<u8>::from(depth_color);
    let depth_rgb = core::Mat::new_rows_cols_with_data(
        h,
        w,
        core::CV_8UC3,
        unsafe { transmute(depth_color.as_ptr()) },
        0,
    )
    .unwrap();

    let mut depth_bgr = depth_rgb.clone().unwrap();

    imgproc::cvt_color(&depth_rgb, &mut depth_bgr, imgproc::COLOR_RGB2BGR, 3).unwrap();

    highgui::imshow(window, &depth_bgr).unwrap();

    let params = opencv::types::VectorOfint::new();
    imgcodecs::imwrite("depth.jpg", &depth_bgr, &params).unwrap();

    highgui::wait_key(10000).unwrap();
}
