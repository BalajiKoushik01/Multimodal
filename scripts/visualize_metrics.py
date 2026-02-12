import matplotlib.pyplot as plt
import re
import os

def parse_log(log_path):
    epochs = []
    losses = []
    
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return epochs, losses

    content = ""
    # Try different encodings
    for encoding in ['utf-8', 'utf-16', 'utf-16le', 'cp1252']:
        try:
            with open(log_path, 'r', encoding=encoding) as f:
                content = f.read()
            break
        except UnicodeError:
            continue
            
    if not content:
        print(f"Could not read {log_path} with standard encodings.")
        return epochs, losses

    for line in content.splitlines():
        # Look for lines like "Epoch 1 Average Loss: 2.1234"
        match = re.search(r"Epoch (\d+) Average Loss: ([\d.]+)", line)
        if match:
            epochs.append(int(match.group(1)))
            losses.append(float(match.group(2)))
    
    return epochs, losses

def plot_loss(epochs, losses, title, filename):
    if not epochs:
        print(f"No data to plot for {title}")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, label='Training Loss', marker='o')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    save_path = os.path.join('visualizations', filename)
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()

if __name__ == "__main__":
    # Ensure directory exists
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')

    # SSL Loss
    ssl_epochs, ssl_losses = parse_log('train_final.log')
    plot_loss(ssl_epochs, ssl_losses, 'Self-Supervised Learning (SSL) Loss', 'ssl_loss.png')

    # Finetune Loss
    ft_epochs, ft_losses = parse_log('finetune.log')
    plot_loss(ft_epochs, ft_losses, 'Fine-Tuning Loss', 'finetune_loss.png')

    # Resume Loss (Latest)
    ft2_epochs, ft2_losses = parse_log('finetune_resume_2.log')
    plot_loss(ft2_epochs, ft2_losses, 'Fine-Tuning Loss (Resume)', 'finetune_resume_2_loss.png')

    # Combined Plot if both exist
    if ssl_epochs and ft_epochs:
        plt.figure(figsize=(12, 6))
        plt.plot(ssl_epochs, ssl_losses, label='SSL Pre-training', alpha=0.7)
        # Shift finetune epochs to follow SSL
        ft_epochs_shifted = [e + max(ssl_epochs) for e in ft_epochs]
        plt.plot(ft_epochs_shifted, ft_losses, label='Fine-Tuning', alpha=0.7)
        
        plt.title('Overall Training Loss Trajectory')
        plt.xlabel('Cumulative Epoch')
        plt.ylabel('Loss')
        plt.axvline(x=max(ssl_epochs), color='r', linestyle='--', label='Phase Switch')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join('visualizations', 'combined_loss.png'))
        print("Saved combined plot to visualizations/combined_loss.png")
