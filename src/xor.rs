use candle_core::{DType, Device, Tensor};
use candle_nn::{linear, Linear, Module, Optimizer, VarBuilder, VarMap, SGD};
use anyhow::Result;

struct XORModel { // MLP Multi Layer Perceptron
    layer1: Linear, 
    layer2: Linear,
    layer3: Linear,
}

impl XORModel {
    fn new(vb: VarBuilder) -> Result<Self> {
        let layer1 = linear(2, 4, vb.pp("layer1"))?;
        let layer2 = linear(4, 8, vb.pp("layer2"))?;
        let layer3 = linear(8, 1, vb.pp("layer3"))?;

        Ok(Self {
            layer1,
            layer2,
            layer3,
        })
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let out = self.layer1.forward(input)?.tanh()?;
        let out = self.layer2.forward(&out)?.tanh()?;
        let out = self.layer3.forward(&out)?.tanh()?;

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
    let model = XORModel::new(vb)?;
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
