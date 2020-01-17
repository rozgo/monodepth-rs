use opencv::{core, highgui, imgproc, imgcodecs, prelude::*};
use std::mem::transmute;
use tch::Device;

fn label_map() -> tch::Tensor {
    //         name               id    color
    // Label(  'road'           ,  0 ,  (128,  64, 128) ),
    // Label(  'sidewalk'       ,  1 ,  (244,  35, 232) ),
    // Label(  'building'       ,  2 ,  ( 70,  70,  70) ),
    // Label(  'wall'           ,  3 ,  (102, 102, 156) ),
    // Label(  'fence'          ,  4 ,  (190, 153, 153) ),
    // Label(  'pole'           ,  5 ,  (153, 153, 153) ),
    // Label(  'traffic light'  ,  6 ,  (250, 170,  30) ),
    // Label(  'traffic sign'   ,  7 ,  (220, 220,   0) ),
    // Label(  'vegetation'     ,  8 ,  (107, 142,  35) ),
    // Label(  'terrain'        ,  9 ,  (152, 251, 152) ),
    // Label(  'sky'            , 10 ,  ( 70, 130, 180) ),
    // Label(  'person'         , 11 ,  (220,  20,  60) ),
    // Label(  'rider'          , 12 ,  (255,   0,   0) ),
    // Label(  'car'            , 13 ,  (  0,   0, 142) ),
    // Label(  'truck'          , 14 ,  (  0,   0,  70) ),
    // Label(  'bus'            , 15 ,  (  0,  60, 100) ),
    // Label(  'train'          , 16 ,  (  0,  80, 100) ),
    // Label(  'motorcycle'     , 17 ,  (  0,   0, 230) ),
    // Label(  'bicycle'        , 18 ,  (119,  11,  32) ),

    let mut labels = vec![vec![30, 15, 60]; 19];
    labels[ 0] = vec![128,  64, 128]; // road
    labels[ 1] = vec![244,  35, 232]; // sidewalk
    labels[ 2] = vec![ 70,  70,  70]; // building
    labels[11] = vec![220,  20,  60]; // person
    labels[12] = vec![255,   0,   0]; // rider
    labels[13] = vec![  0,   0, 142]; // car
    labels[14] = vec![  0,   0,  70]; // truck
    labels[15] = vec![  0,  60, 100]; // bus
    labels[16] = vec![  0,  80, 100]; // train
    labels[17] = vec![  0,   0, 230]; // motorcycle
    labels[18] = vec![119,  11,  32]; // bicycle
    let labels = labels.into_iter().flatten().collect::<Vec<u8>>();
    tch::Tensor::of_slice(&labels)
    .reshape(&[19, 1, 3])
    .permute(&[2, 1, 0])
}

fn main() {
    println!("Start Semantic Segmentation Prediction!");

    let window = "Semantic Segmentation Prediction";
    highgui::named_window(window, 1).unwrap();

    let color_map = label_map().to_device(tch::Device::Cuda(0));
    println!("color_map: {:?}", color_map);

    let img = tch::vision::imagenet::load_image("assets/test_image_04.png").unwrap();
    println!("img: {:?}", img);
    let img = img.unsqueeze(0).to_device(Device::Cuda(0));

    let semnseg_model = tch::CModule::load("models/semseg.pt").unwrap();

    let w = 640;
    let h = 192;

    let semseg_pred = semnseg_model.forward_is(&[tch::IValue::Tensor(img)]).unwrap();
    let semseg_pred = if let tch::IValue::Tensor(semseg_pred) = &semseg_pred {
            Some(semseg_pred)
        } else { None };
    // println!("color_map: {:?}", self.color_map);
    println!("semseg_pred: {:?}", semseg_pred);
    let semseg_pred = semseg_pred.unwrap().squeeze(); 
    let semseg_pred = semseg_pred.argmax(0, false).to_kind(tch::Kind::Uint8);
    println!("semseg_pred: {:?}", semseg_pred);
    
    let color_index = semseg_pred
        .flatten(0, 1)
        .to_kind(tch::Kind::Int64);

    let semseg_color = color_map
        .index_select(2, &color_index)
        .permute(&[2, 1, 0])
        .to_device(tch::Device::Cpu);
    println!("semseg_color: {:?}", semseg_color);

    let semseg_color = Vec::<u8>::from(semseg_color);
    let semseg_rgb = core::Mat::new_rows_cols_with_data(
        h,
        w,
        core::CV_8UC3,
        unsafe { transmute(semseg_color.as_ptr()) },
        0,
    )
    .unwrap();

    let mut semseg_bgr = semseg_rgb.clone().unwrap();

    imgproc::cvt_color(&semseg_rgb, &mut semseg_bgr, imgproc::COLOR_RGB2BGR, 3).unwrap();

    highgui::imshow(window, &semseg_bgr).unwrap();

    let params = opencv::types::VectorOfint::new();
    imgcodecs::imwrite("semseg.jpg", &semseg_bgr, &params).unwrap();

    highgui::wait_key(10000).unwrap();
}
