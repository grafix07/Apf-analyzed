from __future__ import annotations
import struct
import numpy as np
from PIL import Image

ANDROID_FMTS={6407,6408,36196,37492,37494,37496,37497}

def _read_format(data):
    # BBR2 Android texture headers occur in two closely related layouts in
    # extracted packages. In both, the useful 10-u32 record contains:
    # prefix, kind, format, type, width, height, mipCount, payloadSize...
    # Try both known record offsets before falling back to the legacy 42-byte
    # Vector texture header.
    for off in (8, 2):
        if len(data) >= off + 40:
            try:
                v=struct.unpack_from('<10I',data,off)
                fmt=int(v[2])
                if fmt in ANDROID_FMTS or fmt in {1,3,5,16,35,36}:
                    return fmt,int(v[3]),int(v[4]),int(v[5]),max(1,int(v[6]))
            except Exception:
                pass
    if len(data)>=42:
        fmt=struct.unpack_from('<H',data,0x16)[0]
        if fmt in {1,3,5,17,18,19}:
            return fmt,0,struct.unpack_from('<H',data,0x1A)[0],struct.unpack_from('<H',data,0x1E)[0],max(1,struct.unpack_from('<H',data,0x22)[0])
    raise ValueError('Unrecognized Vector texture BIN header/format.')

def _etcpak():
    try: import etcpak; return etcpak
    except Exception as e: raise RuntimeError('ETC/ETC2 skin encoding requires the etcpak package.') from e

def _mips(rgba,count,w,h):
    base=Image.fromarray(rgba,'RGBA')
    for level in range(count):
        mw=max(1,w>>level); mh=max(1,h>>level)
        im=base if level==0 and base.size==(mw,mh) else base.resize((mw,mh),Image.Resampling.LANCZOS)
        a=np.flipud(np.asarray(im.convert('RGBA'),dtype=np.uint8).copy())
        yield a,mw,mh

def _pad4(a):
    h,w=a.shape[:2]; pw=((w+3)//4)*4; ph=((h+3)//4)*4
    if (w,h)==(pw,ph): return a
    out=np.zeros((ph,pw,4),dtype=np.uint8); out[:h,:w]=a; return out

def _etc(a,fmt):
    e=_etcpak(); a=_pad4(a); h,w=a.shape[:2]
    # etcpak expects BGRA input for ETC encoders.
    a=a[:,:, [2,1,0,3]]
    raw=a.tobytes()
    if fmt in (36196,): return bytes(e.compress_etc1_rgb(raw,w,h))
    if fmt in (37492,16): return bytes(e.compress_etc2_rgb(raw,w,h))
    if fmt in (37494,37496,37497,35,36): return bytes(e.compress_etc2_rgba(raw,w,h))
    raise ValueError(f'Unsupported ETC format {fmt}.')

def encode_texture_data(rgba,template):
    rgba=np.asarray(rgba,dtype=np.uint8)
    if rgba.ndim!=3 or rgba.shape[2]!=4: raise ValueError('Custom skin texture must be RGBA8.')
    fmt,typ,ow,oh,mips=_read_format(template); w,h=rgba.shape[1],rgba.shape[0]
    if ow and oh and (w,h)!=(ow,oh):
        rgba=np.asarray(Image.fromarray(rgba,'RGBA').resize((ow,oh),Image.Resampling.LANCZOS),dtype=np.uint8); w,h=ow,oh
    if len(template)>=70 and fmt in ANDROID_FMTS:
        hdr=bytearray(template[:70]); payload=[]
        for a,mw,mh in _mips(rgba,mips,w,h):
            if fmt==6408: payload.append(a.tobytes())
            elif fmt==6407: payload.append(a[:,:,:3].tobytes())
            else: payload.append(_etc(a,fmt))
        blob=b''.join(payload); vals=list(struct.unpack_from('<10I',hdr,8)); vals[4:8]=[w,h,mips,len(blob)]; struct.pack_into('<10I',hdr,8,*vals)
        return bytes(hdr)+blob
    # Desktop BBR2 textures use a 65-byte header with DXGI format 28.
    if len(template)>=65:
        try:
            dfmt=struct.unpack_from('<I',template,22)[0]
            if dfmt==28:
                hdr=bytearray(template[:65]); payload=[]
                for a,mw,mh in _mips(rgba,mips,w,h):
                    payload.append(a.tobytes())
                blob=b''.join(payload)
                struct.pack_into('<I',hdr,26,w); struct.pack_into('<I',hdr,30,h)
                struct.pack_into('<I',hdr,34,mips); struct.pack_into('<I',hdr,38,len(blob))
                return bytes(hdr)+blob
        except Exception:
            pass
    hdr=bytearray(template[:42]); fmt=struct.unpack_from('<H',hdr,0x16)[0]
    if fmt not in {5,17,18,19}: raise ValueError(f'Unsupported Vector texture format {fmt}.')
    e=_etcpak(); payload=[]
    for a,mw,mh in _mips(rgba,mips,w,h):
        if fmt==5: payload.append(a.tobytes())
        elif fmt in (17,18): payload.append(bytes(e.compress_to_dxt1(_pad4(a).tobytes(),_pad4(a).shape[1],_pad4(a).shape[0])))
        else: payload.append(bytes(e.compress_to_dxt5(_pad4(a).tobytes(),_pad4(a).shape[1],_pad4(a).shape[0])))
    blob=b''.join(payload); struct.pack_into('<H',hdr,0x1A,w); struct.pack_into('<H',hdr,0x1E,h); struct.pack_into('<H',hdr,0x22,mips); struct.pack_into('<I',hdr,0x26,len(blob)); return bytes(hdr)+blob
