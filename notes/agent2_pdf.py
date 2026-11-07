"""Minimal ReportLab PDF text extractor (stdlib only): ASCII85 + Flate streams, Tj/TJ strings."""
import re, zlib, base64
def pdf_text(path):
    raw=open(path,'rb').read(); out=[]
    for mm in re.finditer(rb'<<(.*?)>>\s*stream\r?\n(.*?)endstream',raw,re.S):
        d,data=mm.group(1),mm.group(2)
        try:
            if b'ASCII85' in d:
                data=data.strip()
                if data.endswith(b'~>'): data=data[:-2]
                data=base64.a85decode(data)
            if b'Flate' in d: data=zlib.decompress(data)
        except Exception: continue
        lines=[]
        for tj in re.finditer(rb'\((.*?)(?<!\\)\)\s*Tj|\[(.*?)\]\s*TJ|(T\*|Td|TD|ET)',data,re.S):
            if tj.group(1) is not None: lines.append(tj.group(1))
            elif tj.group(2) is not None: lines.append(b''.join(re.findall(rb'\((.*?)(?<!\\)\)',tj.group(2))))
            else: lines.append(b'\n')
        out.append(b' '.join(lines).replace(b'\\(',b'(').replace(b'\\)',b')').decode('latin-1'))
    return '\n'.join(out)
if __name__=='__main__':
    import sys; print(pdf_text(sys.argv[1])[:1500])
