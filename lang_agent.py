'''
Agentes:
1) Estratégia geral:
    - Input: Brand guidance, assets manifest, prompt
    - Output: detailed briefing json: para cada slide: heading, body, suggested_assets, image_briefing, video_briefing

2) Prototyping loop: 

2.1) Detailer: 
    - Input: Detailed briefing
    - Output: Detailed specs: para cada slide: text, assets, image, video: position, size, duration, even more detailed briefing

2.2) Prototyper: 
    - Input: Detailed specs, asset and font files
    - Output: Slides (low res)

2.3) Evaluator: 
    - Input: Slides (low res), Detailed specs
    - Output: Detailed feedback on each slide - new detailed specs

3) Refiner: 
    - Input: Slides (low res), asset and font files, Detailed specs
    - Output: Slides (high res)

4) END