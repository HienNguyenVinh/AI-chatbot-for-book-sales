import pandas as pd

def load_data():
    df = pd.read_csv('\data\book_data.csv')

    def clean_description(description):
        if pd.isna(description):
            return description
        price_description = 'Giá sản phẩm trên Tiki đã bao gồm thuế theo luật hiện hành. Bên cạnh đó, tuỳ vào loại sản phẩm, hình thức và địa chỉ giao hàng mà có thể phát sinh thêm chi phí khác như phí vận chuyển, phụ phí hàng cồng kềnh, thuế nhập khẩu (đối với đơn hàng giao từ nước ngoài có giá trị trên 1 triệu đồng).....'
        if price_description in description:
            description = description.replace(price_description, '')

        return description.replace('\n', ' ').replace('\xa0', ' ').strip()

    def clean_highlight(highlight):
        if pd.isna(highlight):
            return ''
        highlight_list = eval(highlight)
        if isinstance(highlight_list, list):
            return ' '.join(highlight_list)

    df['description'] = df['description'].apply(clean_description)
    df['highlight'] = df['highlight'].apply(clean_highlight)

    df['full_description'] = df['highlight'] + '. ' + df['description']

    return df
