use candle_core::{DType, Device, Tensor};
use candle_nn::{
    conv2d, Conv2d, Conv2dConfig,
    linear, Linear,
    Module,
    VarBuilder, VarMap,
    AdamW, Optimizer,
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
    pixels: usize,
    decode_channels: usize,
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
    fn new(vb: VarBuilder, pixels: usize) -> Result<Self> {
        let half = pixels / 4;
        let channels = 3;
        let decode_channels = 32;
        let encoder_channels = 32;
        let latent = 64;
        let hidden = half * half * decode_channels;
        let encoded = encoder_channels * (pixels / 2) * (pixels / 2);
        let config: Conv2dConfig = Conv2dConfig { stride: 2, padding: 1, ..Default::default() };
        let decode_config: Conv2dConfig = Conv2dConfig { stride: 1, padding: 1, ..Default::default() };
        let encoder: Conv2d = conv2d(channels, encoder_channels, 3, config, vb.pp("conv2d-encoder"))?;
        let fc_mu = linear(encoded, latent, vb.pp("fc_mu"))?;
        let fc_var = linear(encoded, latent, vb.pp("fc_var"))?;
        let encoder_to_decoder = linear(latent, hidden, vb.pp("translator"))?;
        let decoder: Conv2d = conv2d(decode_channels, decode_channels, 3, decode_config, vb.pp("conv2d-decoder"))?;
        let output: Conv2d = conv2d(decode_channels, channels, 3, decode_config, vb.pp("conv2d-output"))?;

        Ok(Self {
            pixels,
            decode_channels,
            encoder,
            decoder,
            encoder_to_decoder,
            fc_mu,
            fc_var,
            output,
        })
    }

    fn reparamerterize(&self, fc_mu: &Tensor, fc_logvar: &Tensor) -> Result<Tensor> {
        let std = (0.5 * fc_logvar)?.exp()?;
        let eps = std.randn_like(0.0, 1.0)?;
        let out = (eps * std)?;
        let out = (out + fc_mu)?;
        Ok(out)
    }

    fn forward(&self, input: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let out = self.encoder.forward(input)?;
        let flat = out.flatten_from(1)?;

        let fc_mu = self.fc_mu.forward(&flat)?;
        let fc_logvar = self.fc_var.forward(&flat)?;

        let latent = self.reparamerterize(&fc_mu, &fc_logvar)?;
        let out = self.decode(&latent)?;

        Ok((out, fc_mu, fc_logvar,))
    }

    fn decode(&self, input: &Tensor) -> Result<Tensor> {
        let batch = input.dim(0)?;
        let out = self.encoder_to_decoder.forward(input)?;
        let out = out.tanh()?;
        let out = out.reshape((batch, self.decode_channels, self.pixels / 4, self.pixels / 4))?;
        let out = out.upsample_nearest2d(self.pixels, self.pixels)?;
        let out = self.decoder.forward(&out)?;
        let out = out.relu()?;
        let out = self.output.forward(&out)?;
        let out = candle_nn::ops::sigmoid(&out)?;

        Ok(out)
    }
}

fn save_image(
    data: &Tensor,
    width: u32,
    height: u32,
    filename: &str,
) -> Result<()> {
    let pixels: Vec<f64> = data
        .permute((0, 2, 3, 1))?
        .contiguous()?
        .flatten_all()?
        .to_vec1()?;

    let buffer: Vec<u8> = pixels
        .into_iter()
        .map( |color| (color.clamp(0.0, 1.0) * 255.0) as u8)
        .collect();

    image::save_buffer(
        filename,
        &buffer,
        width,
        height,
        image::ExtendedColorType::Rgb8,
    )?;

    Ok(())
}

fn load_image(device: &Device, filename: &str, resize: u32) -> Result<Tensor> {
    let img = image::open(filename)?;
    let resized = img.resize_exact(resize, resize, image::imageops::FilterType::Triangle);
    let bytes = resized.to_rgb8().into_raw();
    let pixels: Vec<f64> = bytes
        .into_iter()
        .map( |p| p as f64 / 255.0 )
        .collect();
    let out = Tensor::from_vec(pixels, (1, resize as usize, resize as usize, 3), device)?
        .permute((0, 3, 1, 2))?
        .contiguous()?;
    Ok(out)
}

fn loss_fn(
    output: &Tensor,
    target: &Tensor,
    kld_weight: f64,
    fc_mu: &Tensor,
    fc_logvar: &Tensor,
) ->  Result<Tensor> {
    // KL = 0.5 * (mu^2 + var - 1 - log(var))
    //   with var = exp(fc_logvar), so log(var) = fc_logvar
    // KL = 0.5 * (mu^2 + exp(fc_logvar) - 1 - fc_logvar)
    let kld_loss = ((fc_mu.sqr()? + fc_logvar.exp()?)? - 1.0)?;
    let kld_loss = (kld_loss - fc_logvar)?;
    let kld_loss = kld_loss.mean_all()?;
    let kld_loss = (0.5 * kld_loss)?;

    let loss = candle_nn::loss::mse(output, target)?;
    let out = (loss + (kld_weight * kld_loss)?)?;

    Ok(out)
}

fn main() -> Result<()> {
    println!("Cat picture generator");
    let device = Device::Cpu;
    let pixels: u32 = 32;
    let cat: Tensor = load_image(&device, "cat.png",  pixels)?;

    //println!("Cat: {:?}", cat.shape());
    //println!("Noise: {:?}", input.shape());
    //let features = [[1.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    //let labels   = [[0.0],      [0.0],      [1.0],      [1.0]     ];
    //let input: Tensor = Tensor::new(&features, &device)?; 
    //let targets: Tensor = Tensor::new(&labels, &device)?; 
    //println!("Cat: {:?}", cat.shape());

    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F64, &device);
    let model = VAE::new(vb, pixels as usize)?;
    let kld_weight: f64 = 0.003;
    let (output, fc_mu, fc_logvar) = model.forward(&cat)?;
    let loss = loss_fn(&output, &cat, kld_weight, &fc_mu, &fc_logvar)?;
    println!("Initial Loss: {}", loss.to_scalar::<f64>()?);
    save_image(
        &output,
        pixels,
        pixels,
        "out.png",
    )?;
    //println!("{}", output);

    //println!("Encoder: {:?}", model.encoder.shape());
    
    // Training Phase
    let learning_rate = 0.005;
    let adam_config = candle_nn::ParamsAdamW {
        lr: learning_rate,
        ..Default::default()
    };
    let mut optimizer = AdamW::new(vm.all_vars(), adam_config)?;
    let epochs = 800;
    for epoch in 0..=epochs {
        let (output, fc_mu, fc_logvar) = model.forward(&cat)?;
        let loss = loss_fn(&output, &cat, kld_weight, &fc_mu, &fc_logvar)?;
        optimizer.backward_step(&loss)?;
        let loss_val: f64 = loss.to_scalar()?;
        println!("Epoch: {epoch} Loss: {loss_val:.5}");
    }

    // Re-run the trained model: the `output` above is scoped to the loop.
    let (trained, _fc_mu, _fc_logvar) = model.forward(&cat)?;
    save_image(
        &trained,
        pixels,
        pixels,
        "out-trained.png",
    )?;

    Ok(())
}
