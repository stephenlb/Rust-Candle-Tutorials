use candle_core::{ Device, Tensor };
use candle_nn::{Linear, Module};
use anyhow::Result;

struct XORModel {
    layer1: Linear,
    layer2: Linear,
    layer3: Linear,
}

fn linear(device: &Device, size: (usize, usize)) -> Result<Linear> {
    let weights1 = Tensor::randn(0.0, 1.0, size, device)?;
    let bias1 = Tensor::randn(0f32, 1.0, (100, ), device)?;
    Ok(Linear::new(weights1, Some(bias1)))
}

impl XORModel {
    fn new(device: Device) -> Result<Self> {
        let layer1 = linear(&device, (2, 4))?;
        let layer2 = linear(&device, (4, 8))?;
        let layer3 = linear(&device, (8, 1))?;

        Ok(Self {
            layer1,
            layer2,
            layer3,
        })
    }
    fn forward(&self, input: Tensor) -> Result<Tensor> {
        let out = self.layer1.forward(&input)?;
        //let out = input.matmul(&self.layer1)?;
        //let out = out.matmul(&self.layer2)?;
        //let out = out.matmul(&self.layer3)?;

        Ok(out)
    }
}

fn main() -> Result<()> {
    println!("Stephen's favorite machine learning hello world XOR Operator");
    let device = Device::Cpu;
    let features = [[1.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let labels   = [ 0.0,        0.0,        1.0,        1.0      ];

    let input: Tensor = Tensor::new(&features, &device)?; 
    let model = XORModel::new(device)?;

    let output = model.forward(input)?;

    //println!("{output}");

    Ok(())
}
