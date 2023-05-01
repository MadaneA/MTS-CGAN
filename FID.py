from keras.models import load_model
import numpy as np
import scipy.linalg

model_embedding = load_model("FID_model_embedding_onecondition.h5")

def compute_fid(real_ts, syn_ts):

    # real_signal_mean = real_ts.mean()
    # real_signal_std = real_ts.std()
    # real_signal = (real_ts - real_signal_mean)/(real_signal_std)
    real_signal = real_ts.reshape(-1,151,1,3)

    # synthetic_signal_mean = syn_ts.mean()
    # synthetic_signal_std = syn_ts.std()
    # synthetic_signal = (syn_ts - real_signal_mean)/(real_signal_std)
    synthetic_signal = syn_ts.reshape(-1,151,1,3)

    # compute embeddings for real ts
    real_embeddings = model_embedding.predict(real_signal)
    # compute embeddings for generated ts
    generated_embeddings = model_embedding.predict(synthetic_signal)

    # calculate mean and covariance statistics
    mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
    mu2, sigma2 = generated_embeddings.mean(axis=0), np.cov(generated_embeddings,  rowvar=False)
    
    # calculate sum squared difference between means
    ssdiff = mu1 - mu2 #np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

     # calculate score
    fid = (ssdiff.dot(ssdiff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean) # fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


#### usage ####
# fid = compute_fid(real_image_embeddings, generated_image_embeddings)
