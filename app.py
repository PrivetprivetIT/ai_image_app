import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import uuid
from generators import generate_gpu, generate_cpu

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/generated'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = request.form.get('prompt')
        device = request.form.get('device')  # 'cpu' или 'gpu'

        if not prompt:
            return "Промпт не может быть пустым", 400

        # Уникальное имя файла
        filename = f"{uuid.uuid4().hex}.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            if device == 'gpu':
                generate_gpu(prompt, filepath)
            else:
                generate_cpu(prompt, filepath)
        except Exception as e:
            return f"Ошибка генерации: {str(e)}", 500

        return redirect(url_for('result', filename=filename))

    return render_template('index.html')

@app.route('/result/<filename>')
def result(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return "Файл не найден", 404
    return render_template('result.html', filename=filename)

@app.route('/static/generated/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)