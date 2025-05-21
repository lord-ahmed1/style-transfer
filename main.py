from flask import Flask, render_template , request , redirect , url_for
from style_transfer.style_transfer import style_transfer

app=Flask(__name__)
@app.route('/', methods=['GET','POST'])
def api():  
    if request.method == 'GET':
        return render_template('index.html')
    else:
        style_image = request.files.get("styleImage")
        content_image = request.files.get("contentImage")
        style_intensity = request.form.get("styleIntensity")
        use_transformer = request.form.get("useTransformer") != None
        print('thresh',float(style_intensity)/100)
        if use_transformer:
            model="vit"
        else:
            model="vgg"

        print(model)

        # Save or process files
        if style_image:
            style_image.save("static/uploaded/style/style_uploaded.jpg")
        if content_image:
            content_image.save("static/uploaded/content/content_uploaded.jpg")
        if style_image and content_image:
            style_transfer(style_path="static/uploaded/style/style_uploaded.jpg", content_path="static/uploaded/content/content_uploaded.jpg",output_path="static/uploaded/output/output.jpg", style_threshold=float(style_intensity)/100,num_steps=1000,model_type=model)
        return redirect(url_for('result'))

@app.route('/result')
def result():
    return render_template('result.html')


app.run(host="0.0.0.0",port=1500)
        
