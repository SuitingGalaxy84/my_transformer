import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import ast  # To parse the string representation of lists in your CSV
from collections import Counter

class Vocabulary:
    """
    Simple Vocabulary wrapper to map words to integers.
    """
    def __init__(self, freq_threshold=2):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        """Simple tokenizer that splits by space and lowercases."""
        return [tok.lower().strip(".,;!?") for tok in text.split(" ")]

    def build_vocabulary(self, sentence_list, vocab_pth=None):
        frequencies = Counter()
        idx = 4
        
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

        if vocab_pth:
            with open(vocab_pth, 'wb') as f:
                import pickle
                pickle.dump((self.stoi, self.itos), f)

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

class Flickr30kDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, vocab=None, freq_threshold=2, split='train'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            vocab (Vocabulary, optional): Pre-built vocabulary object.
            freq_threshold (int): Minimum frequency for a word to be included.
            split (string): 'train', 'val', or 'test' to filter the dataset.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Load CSV
        # The CSV has columns: raw, sentids, split, filename, img_id
        df = pd.read_csv(csv_file)
        
        # Filter by split if requested (assuming 'split' column exists based on your snippet)
        if 'split' in df.columns:
            df = df[df['split'] == split]
            
        self.df = df.reset_index(drop=True)

        # Initialize or Load Vocabulary
        self.vocab = vocab
        
        if self.vocab is None:
            self.vocab = Vocabulary(freq_threshold)
            print("Building Vocabulary from scratch...")
            # Extract all captions to build vocab
            all_captions = []
            for raw_list_str in self.df['raw']:
                # The 'raw' column is a string representation of a list: "['cap1', 'cap2']"
                # We use ast.literal_eval to safely parse it into a real python list
                try:
                    captions = ast.literal_eval(raw_list_str)
                    all_captions.extend(captions)
                except:
                    continue # Skip malformed rows
            
            self.vocab.build_vocabulary(all_captions, vocab_pth=None)
            print(f"Vocabulary Size: {len(self.vocab)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # 1. Get Image
        row = self.df.iloc[index]
        img_id = row['filename']
        img_path = os.path.join(self.root_dir, img_id)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            # Create a dummy black image if file is missing (for debugging)
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        # 2. Get Caption
        # The raw column contains a list of 5 captions. 
        # For training, we usually pick one random caption per image to add variety.
        captions_list = ast.literal_eval(row['raw'])
        caption = captions_list[torch.randint(0, len(captions_list), (1,)).item()]

        # 3. Tokenize
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return image, torch.tensor(numericalized_caption)

class MyCollate:
    """
    Custom collate function to pad captions to the same length in a batch.
    """
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        
        targets = [item[1] for item in batch]
        targets = torch.nn.utils.rnn.pad_sequence(
            targets, batch_first=True, padding_value=self.pad_idx
        )

        return imgs, targets

# ==========================================
# Usage Example
# ==========================================
if __name__ == "__main__":
    # Define Transforms (Resize to 224x224 for your ViT)
    transforms_list = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Instantiate Dataset
    # NOTE: Change 'path/to/flickr30k/images' to your actual image folder path
    dataset = Flickr30kDataset(
        csv_file='dataset/flickr_annotations_30k.csv',  # The file you uploaded
        root_dir='dataset/flickr30k-images', 
        transform=transforms_list,
        split='train'
    )

    # Create DataLoader
    pad_idx = dataset.vocab.stoi["<PAD>"]
    loader = DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=MyCollate(pad_idx=pad_idx),
        num_workers=0 # Set to 2 or 4 on Linux/Mac
    )

    # Test one batch
    for images, captions in loader:
        print(f"Batch Image Shape: {images.shape}")   # Should be (4, 3, 224, 224)
        print(f"Batch Caption Shape: {captions.shape}") # Should be (4, Max_Seq_Len)
        print(f"Sample Caption (Indices): {captions[0]}")
        
        # Convert back to text to verify
        decoded = [dataset.vocab.itos[idx.item()] for idx in captions[0] if idx.item() != pad_idx]
        print(f"Decoded: {' '.join(decoded)}")
        break