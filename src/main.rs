use candle_core::{DType, Device, Tensor};
use candle_nn::{
    conv2d, Conv2d, Conv2dConfig,
    // Tensor::max_pool2d
    // Tensor::upsample_nearest2d
    seq, Sequential,
    linear, Linear,
    Module,
    VarBuilder, VarMap,
    SGD, Optimizer
};
use anyhow::Result;


/**
 - TODO build model simple version
 - TODO build full model
 - TODO simple image for training
 - TODO training data
 - TODO image loader png -> Tensor
 - TODO 
 - TODO 
 - TODO 
 - TODO 

**/

struct VAE {
    // TODO The math says you need to add a regulerization term over the output of decoder so it is as close as you can to being normal. so thats the mean squared and the distance of the std from 1 squared
    encoder: Conv2d, 
    // TODO The math says you need to add a regulerization term over the output of decoder so it is as close as you can to being normal. so thats the mean squared and the distance of the std from 1 squared
    encoder_to_decoder: Linear,
    decoder: Conv2d, 
    fc_mu: Linear, 
    fc_var: Linear, 
    output: Conv2d, 
}

/// Variational Auto Encoder
impl VAE {
    fn new(vb: VarBuilder, pixels: usize, hidden: usize, latent: usize) -> Result<Self> {
        let resolution: usize = pixels * pixels * 3;
        let config: Conv2dConfig = Conv2dConfig { padding: 2, ..Default::default() };
        let encoder: Conv2d = conv2d(3, 3, 3, config, vb.pp("conv2d-encoder"))?;
        let encoder_to_decoder = linear(latent, hidden, vb.pp("translator"))?;
        let decoder: Conv2d = conv2d(3, 3, 3, config, vb.pp("conv2d-decoder"))?;
        let fc_mu = linear(hidden, latent, vb.pp("fc_mu"))?;
        let fc_var = linear(hidden, latent, vb.pp("fc_var"))?;
        let output = conv2d(3, 3, 3, config, vb.pp("conv2d-output"))?;

        Ok(Self {
            encoder,
            decoder,
            encoder_to_decoder,
            fc_mu,
            fc_var,
            output,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let out = self.encoder.forward(input)?;
        //let out = self.encoder.forward(input)?;
        //let out = self.layer2.forward(&out)?.tanh()?;
        //let out = self.layer3.forward(&out)?.tanh()?;

        Ok(out)
    }
}

fn load_image(device: &Device, filename: &str, resize: u32) -> Result<Tensor> {
    let img = image::open(filename)?;
    //let resized_img = img.resize(300, 300, FilterType::Lanczos3);
    let resized = img.thumbnail(resize, resize);
    let bytes = resized.to_rgb8().into_raw();
    let pixels: Vec<f64> = bytes
        .into_iter()
        .map( |p| p as f64 / 255.0 )
        .collect();

    let resolution: usize = resize as usize * resize  as usize * 3;
    let out = Tensor::from_vec(pixels, (1, 3, resize as usize, resize as usize), device)?;
    //let out = out.reshape((resize))?;

    Ok(out)
}

fn main() -> Result<()> {
    println!("Cat picture generator");
    let device = Device::Cpu;
    let pixels: u32 = 32;
    let cat: Tensor = load_image(&device, "cat.png",  pixels)?;
    //let features = [[1.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    //let labels   = [[0.0],      [0.0],      [1.0],      [1.0]     ];
    //let input: Tensor = Tensor::new(&features, &device)?; 
    //let targets: Tensor = Tensor::new(&labels, &device)?; 
    println!("Cat: {:?}", cat.shape());

    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F64, &device);
    let model = VAE::new(vb, 32, 64, 32)?;
    let output = model.forward(&cat)?;
    println!("{output}");
    //println!("Encoder: {:?}", model.encoder.shape());
    

    // Training Phase
    /*
    let learing_rate = 0.02;
    let mut optimizer = SGD::new(vm.all_vars(), learing_rate)?;
    let epochs = 800;
    for epoch in 0..=epochs {
        let output = model.forward(&input)?;
        let loss = candle_nn::loss::mse(&output, &targets)?;
        optimizer.backward_step(&loss)?;
        let loss_val: f64 = loss.to_scalar()?;
        println!("Epoch: {epoch} Loss: {loss_val}");
    }
    */

    /*
    let output = model.forward(&input)?;
    println!("{output}");
    */

    Ok(())
}
