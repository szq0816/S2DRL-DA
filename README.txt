# README: Code Instructions for S2DRL-DA Model

## Settings in `main.py`

- **TEST = True**  
  When `TEST = True`, the code directly tests the pre-trained **S2DRL-DA** model without requiring training.  
  When `TEST = False`, the code trains the **S2DRL-DA** model from scratch.  

- **Pre-Trained Results:**  
  We have provided pre-trained monitoring points for each dataset. To reproduce the following results, simply set `TEST = True` and run the code:  

  | Dataset        | ACC     | NMI     | VME     | ARI     | PUR     |
  |----------------|---------|---------|---------|---------|---------|
  | COIL20         | 0.9972  | 0.9954  | 0.9954  | 0.9941  | 0.9972  |
  | FASHION-10K    | 0.9360  | 0.9050  | 0.9050  | 0.8787  | 0.9360  |
  | CIFAR100       | 0.9996  | 0.9994  | 0.9994  | 0.9992  | 0.9996  |
  | MNIST-10K      | 0.9967  | 0.9902  | 0.9902  | 0.9926  | 0.9967  |
  | MNIST_USPS     | 0.9922  | 0.9788  | 0.9788  | 0.9828  | 0.9922  |

---

## Running the Code

To run the code, execute the following command:

```bash
python main.py
```

---

## Notes:

1. **Adjusting the Number of Views:**  
   - If the number of views is 2, modify the import statement in `MvLoad_models.py` as follows:  
     Replace:  
     ```python
     from multi_vae.MvModels3V import VAE
     ```  
     With:  
     ```python
     from multi_vae.MvModels2V import VAE
     ```  

2. **Parameter Optimization:**  
   - Optimal parameters may vary depending on the system configuration. Please adjust accordingly for best results.

--- 

