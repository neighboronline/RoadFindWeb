from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
from .calculate.routeFitting import extract_data_from_excel, Spline_fit_byXY

@csrf_exempt  # 禁用 CSRF 保护，仅用于开发环境，请在生产环境中小心使用
def upload_file(request):
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        file_path = handle_uploaded_file(uploaded_file)
        return JsonResponse({'file_path': file_path})
    else:
        return JsonResponse({'error': 'No file uploaded or invalid request method'}, status=400)

def handle_uploaded_file(uploaded_file):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    upload_dir = os.path.join(base_dir, 'uploaded_files')
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, 'wb+') as destination:
        for chunk in uploaded_file.chunks():
            destination.write(chunk)
    return file_path

@csrf_exempt  # 同样禁用 CSRF 保护
def calculate_data(request):
    if request.method == 'POST':
        file_path = request.POST.get('file_path')
        if file_path:
            data = extract_data_from_excel(file_path)
            zero_segments, non_zero_segments = calculate_segments(data)
            return JsonResponse({
                'zero_segments': zero_segments,
                'non_zero_segments': non_zero_segments
            })
        else:
            return JsonResponse({'error': 'Missing file_path'}, status=400)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)

def calculate_segments(data):
    stake_values = data[:, 1].astype(float)
    length = stake_values[-1] - stake_values[0]
    x_values = data[:, 2].astype(float) 
    y_values = data[:, 3].astype(float)   
    zero_segments, non_zero_segments = Spline_fit_byXY(length, x_values, y_values, show=True)
    return zero_segments, non_zero_segments