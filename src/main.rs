use candle_core::{ Device, Tensor };
use candle_nn::{Linear, Module};
use anyhow::Result;


struct XORModel {
    layer1: Linear,
    layer2: Linear,
    layer3: Linear,
}

impl XORModel {
    fn new(device: Device) -> Result<Self> {
        let weights1 = Tensor::randn(0.0, 1.0, (2, 4), &device)?;
        let bias1 = Tensor::randn(0f32, 1.0, (100, ), &device)?;
        let layer1 = Linear::new(weights1, Some(bias1));

        let weights2 = Tensor::randn(0.0, 1.0, (4, 8), &device)?;
        let bias2 = Tensor::randn(0f32, 1.0, (100, ), &device)?;
        let layer2 = Linear::new(weights2, Some(bias2));

        let weights3 = Tensor::randn(0.0, 1.0, (8, 1), &device)?;
        let bias3 = Tensor::randn(0f32, 1.0, (100, ), &device)?;
        let layer3 = Linear::new(weights3, Some(bias3));

        Ok(Self {
            layer1,
            layer2,
            layer3,
        })
    }
    /*
    fn forward(&self, input: Tensor) -> Result<Tensor> {
        let out = input.matmul(&self.layer1)?;
        let out = out.matmul(&self.layer2)?;
        let out = out.matmul(&self.layer3)?;

        Ok(out)
    }
    */
}

fn main() -> Result<()> {
    println!("Stephen's favorite machine learning hello world XOR Operator");
    let device = Device::Cpu;
    let features = [[1.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let labels   = [ 0.0,        0.0,        1.0,        1.0      ];

    let input: Tensor = Tensor::new(&features, &device)?; 
    let model = XORModel::new(device)?;

    //let output = model.forward(input)?;

    //println!("{output}");

    Ok(())
}
