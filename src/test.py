from preprocess import preprocess_eeg
import logging
import torch
if __name__ == "__main__":
    file = "/home/s20406/data/tuh/export/073_01_tcp_ar_aaaaakyn_s009_t002_preprocessed.pt"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    content = torch.load(file)
    logger.info(content.shape)
    logger.info(content.numpy().shape)

    file2 = "/home/s20406/data/tuh/edf/040/aaaaafxw/s001_2007/02_tcp_le/aaaaafxw_s001_t000.edf"
    processed_data = preprocess_eeg(file2, logger)
    logger.info(processed_data.get_data().shape)