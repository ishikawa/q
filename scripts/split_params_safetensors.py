# 元のファイル名
input_filename = "params.safetensors"  # 実際のファイルパスを指定
# 分割ファイルのプレフィックス
output_prefix = "params_safetensors_"
# 分割サイズ (50MB)
chunk_size = 50 * 1024 * 1024

# ファイルの分割処理
try:
    with open(input_filename, "rb") as infile:
        chunk_index = 0
        while True:
            chunk = infile.read(chunk_size)
            if not chunk:
                break
            output_filename = f"{output_prefix}{chunk_index:03d}"
            with open(output_filename, "wb") as outfile:
                outfile.write(chunk)
            chunk_index += 1
    print("ファイルの分割が完了しました。")
except Exception as e:
    print(f"エラーが発生しました: {e}")
