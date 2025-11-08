import requests
import json

# RxNorm API 的基本 URL
RXNORM_API_URL = "https://rxnav.nlm.nih.gov/REST"

# 药品列表
medications = [
    'Acyclovir',
    'Acyclovir',
    'Adalat',
    'Alinamin',
    'Allopurinol',
    'Ambroxol',
    'Amdixal',
    'Amdixal',
    'Amitriptyline',
    'Amlodipin',
    'Amlodipin',
    'Amoxicillin',
    'Amoxicillin',
    'Anxibloc',
    'Apazol',
    'Apazol',
    'Apidra',
    'Aptor',
    'Arimidex',
    'Asam',
    'Asering',
    'Aspilets',
    'Atorvastatin',
    'Azithromycin',
    'Berotec',
    'Betaserc',
    'Bisoprolol',
    'Braxidin',
    'Cameloc',
    'Cameloc',
    'Canderin',
    'Canderin',
    'Candesartan',
    'Candesartan',
    'Carbloxal',
    'Cardace',
    'Cefadroxil',
    'Cefadroxil',
    'Cefixime',
    'Cefixime',
    'Cefotaxime',
    'Ceftazidime',
    'Ceftriaxone',
    'Cendo',
    'Cetirizine',
    'Cetirizine',
    'Ciprofloxacine',
    'Ciprofloxacine',
    'Clindamycin',
    'Clindamycin',
    'Clonidine',
    'Clopisan',
    'Co',
    'Concor',
    'Concor',
    'Cordila',
    'Decolsin',
    'Deculin',
    'Deculin',
    'Deksametason',
    'Depakote',
    'Dexanta',
    'Dexanta',
    'Diazole',
    'Digoxine',
    'Domperidon',
    'Domperidone',
    'Dorner',
    'Dulcolax',
    'Dulcolax',
    'Eclid',
    'Eclid',
    'Elkana',
    'Elkana',
    'Enystin',
    'Esvat',
    'Euthyrox',
    'Farbivent',
    'Farsorbid',
    'Femara',
    'Fitbon',
    'Fleet',
    'Flunarizine',
    'Fosen',
    'Furosemid',
    'Furosemide',
    'Fusycom',
    'Gabexal',
    'Gabexal',
    'Ganin',
    'Gastrofer',
    'Gentamicin',
    'Gentamycin',
    'Glimepiride',
    'Glimepiride',
    'Glimepiride',
    'Glimepiride',
    'Glucobay',
    'Glucodex',
    'Glurenorm',
    'Harnal',
    'Harnal',
    'Hidroklorotiazid',
    'Hydrocortison',
    'Hyoscine',
    'Hystolan',
    'Hytrin',
    'Hytrin',
    'Hytroz',
    'Ibuprofen',
    'Induxin',
    'Inerson',
    'Irbedox',
    'Irvebal',
    'Irvebal',
    'Isosorbid',
    'KA',
    'Kalnex',
    'Kalnex',
    'Kalnex',
    'Kaltrofen',
    'Kamadol',
    'Ketokonazol',
    'Ketokonazol',
    'Ketoprofen',
    'Ketorolac',
    'Kotrimoksazol',
    'KSR',
    'Lansoprazole',
    'Lantus',
    'Laxadine',
    'Levemir',
    'Levofloxacin',
    'Loratadine',
    'Lovenox',
    'Madopar',
    'Meloxicam',
    'Meloxicam',
    'Metformin',
    'Methylprednisolone',
    'Metronidazole',
    'Micardis',
    'Miconazol',
    'Mucogard',
    'Neo-Mercazole',
    'Neurodex',
    'New',
    'Nifedipin',
    'Nilacol',
    'Nitrokaf',
    'Nitrokaf',
    'Noperten',
    'Norelut',
    'Norvask',
    'Novalgin',
    'Novo',
    'Novo',
    'Olmetec',
    'Omeprazole',
    'Ondansetron',
    'Ondansetron',
    'Ondansetron',
    'Osteocal',
    'Otsu',
    'Otsu',
    'Otsu',
    'Paracetamol',
    'Paracetamol',
    'Paracetamol',
    'Paratusin',
    'Paratusin',
    'Phenytoin',
    'Propiltiourasil',
    'Propranolol',
    'Pulmicort',
    'Pyrazinamide',
    'Ramixal',
    'Ranitidine',
    'Ranitidine',
    'Renadinac',
    'Retaphyl',
    'Rhinofed',
    'Rifampicin',
    'Ringer',
    'Salbutamol',
    'Salofalk',
    'Scabimite',
    'Scobutrin',
    'Simarc',
    'Simvastatin',
    'Simvastatin',
    'Sod.',
    'Sod.',
    'Sofra-Tulle',
    'Sohobion',
    'Solosa',
    'Solosa',
    'Solosa',
    'Sotatic',
    'Sotatic',
    'Spironolacton',
    'Spironolacton',
    'Starfolat',
    'Stesolid',
    'Symbicort',
    'Tetagam',
    'Thyrozol',
    'Tiaryt',
    'Tramadol',
    'Triaxitrol',
    'Triheksifenidil',
    'Trolip',
    'Ulsafate',
    'Ulsidex',
    'Urixin',
    'Vaclo',
    'Vagistin',
    'Valdimex',
    'Vastigo',
    'Vbloc',
    'Ventolin',
    'Ventolin',
    'Voltadex',
    'Xanvit',
    'Zink',
    'Zinkid',
    'Dilavask',
    'Platogrix',
    'Nitral',
    'Lyrica',
    'Erythromycin',
    'Zilop',
    'Kalium',
    'Allopurinol',
    'Oralit',
    'Acyclovir',
    'Ambroxol',
    'Aminefron',
    'Amoxsan',
    'Amoxsan',
    'Antasida',
    'Apialys',
    'Asam',
    'ATS',
    'Baquinor',
    'Becom',
    'Betadin',
    'Biodiar',
    'BREATHY',
    'Buscopan',
    'Carbol',
    'Cefat',
    'Cefat',
    'CEFXON',
    'Cendo',
    'CENDO',
    'Chlorpeniramine',
    'CITICOLINE',
    'CLANEKSI',
    'Clobazam',
    'Codein',
    'Dexamethason',
    'Dexamethason',
    'Dextamin',
    'Dextrose',
    'Dumin',
    'Dumin',
    'Elpicef',
    'ENGERIX',
    'EPERISONE',
    'Epexol',
    'Epexol',
    'Equal',
    'Extrace',
    'FENTANIL',
    'FG',
    'FIXACEP',
    'FLU',
    'Gentamicine',
    'Gliseril',
    'GRACEF',
    'Imboost',
    'Imunos',
    'Imunos',
    'Infusan',
    'Inpepsa',
    'Ka-EN',
    'Kandistatin',
    'Kapsul',
    'Kapsul',
    'kapsul',
    'Ketosteril',
    'Ketricin',
    'Lameson',
    'Lanolin',
    'Lapicef',
    'Lapicef',
    'Lasgan',
    'LCD',
    'Lidocain',
    'Lycoxy',
    'Maltiron',
    'MECOBALAMIN',
    'MECOBALAMIN',
    'Mefinal',
    'Metronidazole',
    'Milmor',
    'Mucopect',
    'NaCl',
    'Naprex',
    'Neurosanbe',
    'Neurosanbe',
    'Nonflamin',
    'OBH',
    'Pehacain',
    'PRAZOTEC',
    'Proneuron',
    'RHINOS',
    'RHINOS',
    'Sanadryl',
    'Sanadryl',
    'Sanmol',
    'Sanmol',
    'Sanmol',
    'Sirdalud',
    'STESOLID',
    'Syntocinon',
    'Terfacef',
    'Thrombophob',
    'Tiriz',
    'Tremenza',
    'Tricefin',
    'Tuzalos',
    'UNALIUM',
    'Verorab',
    'Viccillin',
    'VITAMIN',
    'Vitazym',
    'Vometa',
    'Vometa',
    'Xylocain',
    'Acid',
    'Alkohol',
    'CHROMALUX',
    'Durogesic',
    'EPISAN',
    'FARMADOL',
    'ISPRINOL',
    'LASAL',
    'NEURALGIN',
    'NOCID',
    'TRAMSET',
    'ZINCPRO',
    'Angiocath',
    'Angiocath',
    'Angiocath',
    'IV',
    'Micro',
    'Opsite',
    'Pot',
    'Stomach',
    'Suction',
    'Transfusi',
    'T-VIO',
    'Urine',
    'l-',
    'Disp',
    'Disp',
    'Disp',
    'Folley',
    'Kasa',
    'Mucos',
    'Vitamin',
    '3',
    'Mometasone',
    'Molexdryl',
    'Ramipril',
    'Ramipril',
    'citicolin',
]


# 查询 RxNorm API，获取药品的 RxCUI（RxNorm 唯一标识符）
def get_rxcui(drug_name):
    search_url = f"{RXNORM_API_URL}/search?name={drug_name}&maxReturn=1"
    response = requests.get(search_url)
    if response.status_code == 200:
        data = response.json()
        if data['idGroup']['rxnormId']:
            return data['idGroup']['rxnormId'][0]  # 返回第一个匹配的 RxCUI
    return None


# 查询 RxNorm API 获取替代药物
def get_alternate_drugs(rxcui):
    alternate_url = f"{RXNORM_API_URL}/rxcui/{rxcui}/related?rela=SY Non-Proprietary"
    response = requests.get(alternate_url)
    if response.status_code == 200:
        data = response.json()
        if 'relatedGroup' in data:
            return [item['rxcui'] for item in data['relatedGroup']['conceptGroup']]
    return []


# 查询 RxNorm API 获取组合药物
def get_combination_drugs(rxcui):
    combination_url = f"{RXNORM_API_URL}/rxcui/{rxcui}/related?rela=Has_Ingredient"
    response = requests.get(combination_url)
    if response.status_code == 200:
        data = response.json()
        if 'relatedGroup' in data:
            return [item['rxcui'] for item in data['relatedGroup']['conceptGroup']]
    return []


import requests
import pandas as pd

# DrugBank API 的基本 URL
DRUGBANK_API_URL = "https://api.drugbank.com/v1/drugs"

# 设置 API 密钥（从 DrugBank 网站获得）
API_KEY = "YOUR_API_KEY_HERE"



# 函数：查询 DrugBank API 获取药品的信息
def get_drug_info(drug_name):
    headers = {
        'Authorization': f'Bearer {API_KEY}'
    }
    # 通过药品名称查询 DrugBank 数据库
    response = requests.get(f'{DRUGBANK_API_URL}/{drug_name}', headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return None


# 生成三元组并存储
triplets = []

# 查询每个药品的信息
for drug in medications:
    print(f"查询药品: {drug}")
    drug_info = get_drug_info(drug)

    if drug_info:
        try:
            # 处理返回的药品数据
            brand_name = drug_info.get('name', 'Unknown')
            substitutes = drug_info.get('substitutes', [])
            combinations = drug_info.get('combinations', [])

            # 处理替代药物
            for substitute in substitutes:
                triplets.append((brand_name, '替代', substitute))

            # 处理组合药物
            for combination in combinations:
                triplets.append((brand_name, '组合', combination))

        except KeyError as e:
            print(f"错误处理药品 {drug}: {e}")
    else:
        print(f"未找到药品 {drug} 的信息。")

# 将结果保存到 CSV 文件
df = pd.DataFrame(triplets, columns=['药品', '关系', '相关药品'])
df.to_csv('medication_triplets.csv', index=False, encoding='utf-8')

print("三元组数据已保存到 medication_triplets.csv")
