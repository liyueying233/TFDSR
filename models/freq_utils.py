import torch

def fftSplitAP(x):
    # spatial domain 2 frequency domain
    x_freq = torch.fft.fftshift(torch.fft.fftn(x, dim=(-2, -1)), dim=(-2, -1))
    
    # compute A P
    A_x = torch.abs(x_freq)
    P_x = torch.angle(x_freq)
    
    return A_x, P_x


def fftCombineAP(A_x, P_x):
    # Combine the real and imaginary parts to form the complex frequency representation
    F_prime = torch.polar(A_x.float(), P_x.float())
    
    # Perform the inverse FFT shift and inverse FFT
    x_reconstructed = torch.fft.ifftn(torch.fft.ifftshift(F_prime, dim=(-2, -1)), dim=(-2, -1)).real
    
    return x_reconstructed