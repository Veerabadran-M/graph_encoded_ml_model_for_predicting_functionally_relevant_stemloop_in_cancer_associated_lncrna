from utility_functions import print_loading_bar, compute_mfe
from pathlib import Path
import forgi.graph.bulge_graph as fgb
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import pandas as pd
import os

def lncrna_structural_analysis_xml(sequence, structure, filename, attributes=None, mfe=None):

    bg = fgb.BulgeGraph.from_dotbracket(structure, sequence)

    def seq_of(elem):

        parts = bg.get_define_seq_str(elem, adjacent=False)
        if not parts:
            return ""
        joined = []
        for p in parts:
            if isinstance(p, (list, tuple)):
                joined.append("".join(map(str, p)))
            else:
                joined.append(str(p))
        return "".join(joined)

    root = ET.Element("RNAStructureAnalysis", attributes)
    ET.SubElement(root, "Sequence", length=str(len(sequence))).text = sequence
    ET.SubElement(root, "Structure", mfe=mfe).text = structure

    stems_el = ET.SubElement(root, "Stems")
    for s in bg.stem_iterator():
        stem_el = ET.SubElement(stems_el, "Stem", id=s)
        bp_list = list(bg.stem_bp_iterator(s))
        side1_pos = [i for i, _ in bp_list]
        side2_pos = [j for _, j in bp_list]
        side1_seq = "".join(sequence[p-1] for p in side1_pos)
        side2_seq = "".join(sequence[p-1] for p in side2_pos[::-1])

        s1_el = ET.SubElement(stem_el, "StemSide", id=f"{s}s1")
        ET.SubElement(s1_el, "Sequence").text = side1_seq
        ET.SubElement(s1_el, "Length").text = str(len(side1_seq))
        ET.SubElement(s1_el, "Position").text = ",".join(map(str, side1_pos))

        s2_el = ET.SubElement(stem_el, "StemSide", id=f"{s}s2")
        ET.SubElement(s2_el, "Sequence").text = side2_seq
        ET.SubElement(s2_el, "Length").text = str(len(side2_seq))
        ET.SubElement(s2_el, "Position").text = ",".join(map(str, side2_pos[::-1]))

        ET.SubElement(stem_el, "Basepairs").text = ";".join(f"({i},{j})" for i,j in bp_list)

    hairpins_el = ET.SubElement(root, "Hairpins")
    for h in bg.hloop_iterator():
        hairpin_el = ET.SubElement(hairpins_el, "Hairpin", id=h)
        hp_seq = seq_of(h)
        pos = list(bg.define_residue_num_iterator(h))
        ET.SubElement(hairpin_el, "Sequence").text = hp_seq
        ET.SubElement(hairpin_el, "Length").text = str(len(hp_seq))
        ET.SubElement(hairpin_el, "Position").text = ",".join(map(str,pos))

    iloops_el = ET.SubElement(root, "Internalloops")
    for i in bg.iloop_iterator():
        il_el = ET.SubElement(iloops_el, "Internalloop", id=i)
        dims = bg.get_bulge_dimensions(i)
        seq_i = "".join(bg.get_define_seq_str(i))
        all_pos = list(bg.define_residue_num_iterator(i))
        side1_pos, side2_pos = all_pos[:dims[0]], all_pos[dims[0]:]
        side1_seq, side2_seq = seq_i[:dims[0]], seq_i[dims[0]:]

        s1_el = ET.SubElement(il_el, "InternalloopSide", id=f"{i}s1")
        ET.SubElement(s1_el, "Sequence").text = side1_seq
        ET.SubElement(s1_el, "Length").text = str(len(side1_seq))
        ET.SubElement(s1_el, "Position").text = ",".join(map(str, side1_pos))

        s2_el = ET.SubElement(il_el, "InternalloopSide", id=f"{i}s2")
        ET.SubElement(s2_el, "Sequence").text = side2_seq
        ET.SubElement(s2_el, "Length").text = str(len(side2_seq))
        ET.SubElement(s2_el, "Position").text = ",".join(map(str, side2_pos))

    mloops_el = ET.SubElement(root, "Multiloops")
    for m in bg.mloop_iterator():
        if m not in bg.defines:
            continue
        ml_el = ET.SubElement(mloops_el, "Multiloop", id=m)
        ml_seq = seq_of(m)
        ml_pos = list(bg.define_residue_num_iterator(m))
        ET.SubElement(ml_el, "Sequence").text = ml_seq
        ET.SubElement(ml_el, "Length").text = str(len(ml_seq))
        ET.SubElement(ml_el, "Position").text = ",".join(map(str,ml_pos))

    floops_el = ET.SubElement(root, "Floops")
    for f in bg.floop_iterator():
        f_el = ET.SubElement(floops_el, "Floop", id=f)
        f_seq = seq_of(f)
        f_pos = list(bg.define_residue_num_iterator(f))
        ET.SubElement(f_el, "Sequence").text = f_seq
        ET.SubElement(f_el, "Length").text = str(len(f_seq))
        ET.SubElement(f_el, "Position").text = ",".join(map(str,f_pos))

    tloops_el = ET.SubElement(root, "Tloops")
    for t in bg.tloop_iterator():
        t_el = ET.SubElement(tloops_el, "Tloop", id=t)
        t_seq = seq_of(t)
        t_pos = list(bg.define_residue_num_iterator(t))
        ET.SubElement(t_el, "Sequence").text = t_seq
        ET.SubElement(t_el, "Length").text = str(len(t_seq))
        ET.SubElement(t_el, "Position").text = ",".join(map(str,t_pos))

    stemloops_el = ET.SubElement(root, "Stemloops")
    for h in bg.hloop_iterator():
        neighbors = list(bg.edges[h])
        stem = next((n for n in neighbors if n.startswith("s")), None)
        if stem:
            sl_el = ET.SubElement(stemloops_el, "Stemloop", id=f"sl{h}")
            ET.SubElement(sl_el, "Stem", ref=stem)
            ET.SubElement(sl_el, "Hairpin", ref=h)

    adj_el = ET.SubElement(root, "Adjacency")
    for elem, nbrs in bg.edges.items():
        e_el = ET.SubElement(adj_el, "Element", id=elem)
        ET.SubElement(e_el, "Neighbors").text = ",".join(nbrs)

    xml_str = ET.tostring(root, encoding='utf-8')
    dom = minidom.parseString(xml_str)
    pretty_xml_as_str = dom.toprettyxml(indent="    ")

    with open(filename, "w", encoding="utf-8") as f:
        f.write(pretty_xml_as_str)

def rna_structural_analysis(base_dir, excel, xml_dir):

    base = Path(base_dir)
    input_path = base / "processed-data" / excel
    folder_path = base/ "processed-data" / xml_dir

    os.makedirs(folder_path, exist_ok=True)

    reference_excel = pd.read_excel(input_path)

    entries = len(reference_excel)

    accession_ids = reference_excel["Accession_ID"]
    symbols = reference_excel["Symbol"]
    gene_ids = reference_excel["Gene_ID"]
    expressions = reference_excel["Expression"]
    sequences = reference_excel["Sequence"]
    structures = reference_excel["Secondary_structure_representation"]
    mfes = reference_excel["MFE"]

    for idx in range(entries):

        accession_id = str(accession_ids[idx])
        symbol = str(symbols[idx])
        gene_id = str(gene_ids[idx])
        expression = str(expressions[idx])
        sequence = str(sequences[idx])
        structure = str(structures[idx])
        mfe = str(mfes[idx])

        print_loading_bar(idx, entries, 50, f"Analyzing {symbol}", "green")

        filename = os.path.join(folder_path, f"{symbol}.xml")

        attributes = {
            "accession_id"  : accession_id,
            "symbol"        : symbol,
            "gene_id"       : gene_id,
            "expression"    : expression
        }

        lncrna_structural_analysis_xml(sequence, structure, filename, attributes, mfe)

        if idx == entries - 1:
            print_loading_bar(entries, entries, 50, f"All lncrna(s) are analyzed and exported as xml file.", "green")

def extract_stemloop_data_from_xml(xml_file):

    stemloop_data = {
        "Accession_ID": [],
        "Symbol": [],
        "Gene_ID": [],
        "Expression": [],
        "Sequence": [],
        "Sequence_length": [],
        "Structure": [],
        "Tot_stemloops": [],
        "Stemloop_id": [],
        "Sequence_of_stemloop": [],
        "Structure_of_stemloop": [],
        "Seq_len_of_stemloop": [],
        "GC_content": [],
        "Stem_length": [],
        "Loop_length": [],
        "Start_end_pos": [],
        "MFE_of_stem_loop": []
        }

    tree = ET.parse(xml_file)
    root = tree.getroot()

    accesion_id = root.attrib.get("accession_id", "")
    symbol = root.attrib.get("symbol", "")
    gene_id = root.attrib.get("gene_id", "")
    expression = root.attrib.get("expression", "")

    sequence = root[0]
    structure = root[1]

    rna_seq = sequence.text.strip()
    rna_seq_len = int(sequence.attrib.get("length", ""))
    rna_structure = structure.text.strip()
    rna_mfe = float(structure.attrib.get("mfe", ""))

    stemloops = root[8]
    no_stemloops = len(stemloops)

    for stemloop in stemloops:

        stemloop_id = stemloop.attrib.get("id", "")

        stem = stemloop[0]
        hairpin = stemloop[1]

        stem_id = stem.attrib.get("ref", "")
        hairpin_id = hairpin.attrib.get("ref", "")

        stem = root.find(f".//Stem[@id='{stem_id}']")
        stem_position = list(map(int, stem[0][2].text.split(","))) + list(map(int, stem[1][2].text.split(",")))
        hairpin = root.find(f".//Stem[@id='{hairpin_id}']")
        hairpin_position = list(map(int, stem[0][2].text.split(",")))

        start_pos, end_pos = stem_position[0] - 1, stem_position[-1] - 1
        stemloop_seq = rna_seq[start_pos:end_pos + 1]
        stemloop_structure = rna_structure[start_pos:end_pos + 1]
        stemloop_len = end_pos - start_pos + 1
        stem_len = len(stem_position)
        hairpin_len = len(hairpin_position)

        stemloop_mfe = compute_mfe(stemloop_seq, stemloop_structure)
        gc_content = int(((stemloop_seq.count("G") + stemloop_seq.count("C")) / stemloop_len) * 100)

        stemloop_data["Accession_ID"].append(accesion_id)
        stemloop_data["Symbol"].append(symbol)
        stemloop_data["Gene_ID"].append(gene_id)
        stemloop_data["Expression"].append(expression)
        stemloop_data["Sequence"].append(rna_seq)
        stemloop_data["Sequence_length"].append(rna_seq_len)
        stemloop_data["Structure"].append(rna_structure)
        stemloop_data["Tot_stemloops"].append(no_stemloops)
        stemloop_data["Stemloop_id"].append(f"{symbol}_{stemloop_id}")
        stemloop_data["Sequence_of_stemloop"].append(stemloop_seq)
        stemloop_data["Seq_len_of_stemloop"].append(stemloop_len)
        stemloop_data["GC_content"].append(gc_content)
        stemloop_data["Stem_length"].append(stem_len)
        stemloop_data["Loop_length"].append(stemloop_len - stem_len)
        stemloop_data["Start_end_pos"].append((start_pos + 1, end_pos + 1))
        stemloop_data["Structure_of_stemloop"].append(stemloop_structure)
        stemloop_data["MFE_of_stem_loop"].append(stemloop_mfe)

    stemloop_df = pd.DataFrame(stemloop_data)
    return stemloop_df

def extract_stemloop_data(base_dir, xml_dir, stemloop_file):

    base = Path(base_dir)
    XML_FOLDER = base / "processed-data" / xml_dir

    stemloop_df = pd.DataFrame()

    count = 0
    total = len(os.listdir(XML_FOLDER))

    for xml_file in os.listdir(XML_FOLDER):
        print_loading_bar(count, total, 50, f"Parsing {xml_file}...")
        stemloop_df = pd.concat([stemloop_df, extract_stemloop_data_from_xml(XML_FOLDER / xml_file)], ignore_index=True)
        count += 1

    print_loading_bar(total, total, 50, f"All files parsed.")

    stemloop_df.to_excel(base / "processed-data" / stemloop_file, index=False)