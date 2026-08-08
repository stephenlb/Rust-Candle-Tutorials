use candle_core::{DType, Device, Tensor};
use candle_nn::{
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
    encoder: Sequential, 
    encoder_to_decoder: Linear,
    decoder: Sequential, 
    fc_mu: Linear, 
    fc_var: Linear, 
}

/// Variational Auto Encoder
impl VAE {
    fn new(vb: VarBuilder, hidden: usize, latent: usize) -> Result<Self> {
        let encoder: Sequential = seq()
            .add(linear(2, hidden, vb.pp("encoder-layer1"))?)
            .add(linear(hidden, hidden, vb.pp("encoder-layer2"))?)
            .add_fn( |xs| xs.tanh() );
        let encoder_to_decoder = linear(latent, hidden, vb.pp("layer3"))?;
        let decoder: Sequential = seq()
            .add(linear(hidden, latent, vb.pp("encoder-layer1"))?)
            .add(linear(hidden, latent, vb.pp("encoder-layer2"))?)
            .add_fn( |xs|  xs.tanh() );
        let fc_mu = linear(2, 4, vb.pp("layer1"))?;
        let fc_var = linear(2, 4, vb.pp("layer2"))?;

        Ok(Self {
            encoder,
            decoder,
            encoder_to_decoder,
            fc_mu,
            fc_var,
        })
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let out = self.encoder.forward(input)?;
        //let out = self.layer2.forward(&out)?.tanh()?;
        //let out = self.layer3.forward(&out)?.tanh()?;

        Ok(out)
    }
}

fn main() -> Result<()> {
    println!("Stephen's favorite machine learning hello world XOR Operator");
    let device = Device::Cpu;
    let features = [[1.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let labels   = [[0.0],      [0.0],      [1.0],      [1.0]     ];
    let input: Tensor = Tensor::new(&features, &device)?; 
    let targets: Tensor = Tensor::new(&labels, &device)?; 

    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F64, &device);
    let model = VAE::new(vb, 64, 32)?;
    let learing_rate = 0.02;
    let mut optimizer = SGD::new(vm.all_vars(), learing_rate)?;

    // Training Phase
    let epochs = 800;
    for epoch in 0..=epochs {
        let output = model.forward(&input)?;
        let loss = candle_nn::loss::mse(&output, &targets)?;
        optimizer.backward_step(&loss)?;
        let loss_val: f64 = loss.to_scalar()?;
        println!("Epoch: {epoch} Loss: {loss_val}");
    }

    let output = model.forward(&input)?;
    println!("{output}");

    Ok(())
}
