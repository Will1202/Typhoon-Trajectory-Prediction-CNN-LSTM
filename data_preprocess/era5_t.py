import pandas as pd
from datetime import datetime, timedelta

def convert_bjt_to_utc(input_file, output_file):
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    # 检查是否存在DateTime(BJT)列
    if 'DateTime(BJT)' not in df.columns:
        print("Error: 'DateTime(BJT)' column not found in the CSV file.")
        return
    
    # 定义转换函数
    def bjt_to_utc(bjt_time):
        try:
            # 将字符串转换为 datetime 对象
            dt = datetime.strptime(str(bjt_time), '%Y%m%d%H')
            # 减去 8 小时转换为 UTC
            dt_utc = dt - timedelta(hours=8)
            # 转换回字符串格式
            return dt_utc.strftime('%Y%m%d%H')
        except ValueError:
            print(f"Error parsing time: {bjt_time}")
            return None
    
    # 应用转换函数
    df['DateTime(UTC)'] = df['DateTime(BJT)'].apply(bjt_to_utc)
    
    # 删除原始的 DateTime(BJT) 列
    df.drop(columns=['DateTime(BJT)'], inplace=True)
    
    # 重命名新列
    df.rename(columns={'DateTime(UTC)': 'DateTime(UTC)'}, inplace=True)
    
    # 保存到新文件
    df.to_csv(output_file, index=False)
    print(f"File successfully saved to {output_file}")

# 使用示例
input_file = '/home/dl392/data/yiwei/typhoon/data_preprocess/best_track_records.csv'  # 输入文件名
output_file = '/home/dl392/data/yiwei/typhoon/data_preprocess/best_track_records.csv_p1.csv'  # 输出文件名
convert_bjt_to_utc(input_file, output_file)
