# rad-metric
Metrics for radiology eval

## Setup

Run the following script to setup environment:

```bash
conda create -n rad python=3.12
conda activate rad
pip3 install 'numpy<2.0.0'
pip3 install torch --index-url https://download.pytorch.org/whl/cu128
pip3 install git+https://github.com/hiaoxui/radgraph
pip3 install -r requirements.txt
```
