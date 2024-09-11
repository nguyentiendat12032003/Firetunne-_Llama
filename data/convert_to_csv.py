from datasets import load_dataset
import pandas as pd

# Tải dataset
ds = load_dataset("hungnm/vietnamese-medical-qa")

# Định nghĩa hàm biến đổi dữ liệu
def transform_conversation(example):
    user_text = example['question'].strip()
    ai_text = example['answer'].strip()

    # Tạo định dạng prompt theo yêu cầu
    formatted_text = (f'<s>[INST] {user_text} [/INST]\n {ai_text} </s>')

    return {'text': formatted_text}

# Áp dụng hàm biến đổi cho dataset
transformed_dataset = ds['train'].map(transform_conversation)

# Loại bỏ các cột không cần thiết
transformed_dataset = transformed_dataset.remove_columns(['question', 'answer'])

# Chuyển dataset thành DataFrame của Pandas để dễ xử lý
df = pd.DataFrame(transformed_dataset)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
train_df = df.sample(frac=0.7, random_state=42)
valid_df = df.drop(train_df.index)

# Lưu dữ liệu thành file CSV
train_df.to_csv("train.csv", index=False)
valid_df.to_csv("valid.csv", index=False)

print("Dữ liệu đã được lưu thành công.")
