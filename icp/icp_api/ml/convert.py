import csv

def convert_csv_to_utf8(input_file_path, output_file_path, input_encoding='cp1252', errors='ignore'):
    """
    Convert a CSV file from Windows-1252 encoding to UTF-8, removing or replacing characters that cannot be converted,
    by reading the file in binary mode and manually decoding each line.

    Args:
    - input_file_path (str): The path to the input CSV file.
    - output_file_path (str): The path to the output CSV file.
    - input_encoding (str, optional): The encoding of the input CSV file. Default is 'windows-1252'.
    - errors (str, optional): The error handling scheme ('ignore' or 'replace').

    Returns:
    - None
    """
    with open(input_file_path, 'rb') as infile, \
         open(output_file_path, mode='w', newline='', encoding='utf-8') as outfile:
        # Create a reader to iterate over lines in binary mode
        reader = csv.reader((line.decode(input_encoding, errors=errors) for line in infile), delimiter=',', quotechar='"')
        writer = csv.writer(outfile)
        
        for row in reader:
            writer.writerow(row)

# Example usage
input_file_path = 'movies.csv'
output_file_path = 'utmovies.csv'
convert_csv_to_utf8(input_file_path, output_file_path)
