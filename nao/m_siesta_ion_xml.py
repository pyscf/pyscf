from __future__ import division
import numpy as np
from xml.dom import minidom
#from ase.calculators.siesta.mbpt_lcao_utils import str2int, str2float
import sys
import re

def str2int(string):
  numeric_const_pattern = r"""
  [-+]? # optional sign
  (?:
    (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
    |
    (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc
  )
  # followed by optional exponent part if desired
  (?: [Ee] [+-]? \d+ ) ?
  """
  rx = re.compile(numeric_const_pattern, re.VERBOSE)
  nb = rx.findall(string)
  for i in enumerate(nb): nb[i[0]] = int(i[1])
  return np.array(nb)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

def str2float(string):
  numeric_const_pattern = r"""
  [-+]? # optional sign
  (?:
    (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
    |
    (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc
  )
  # followed by optional exponent part if desired
  (?: [Ee] [+-]? \d+ ) ?
  """
  rx = re.compile(numeric_const_pattern, re.VERBOSE)
  nb = rx.findall(string)
  for i in enumerate(nb): nb[i[0]] = float(i[1])

  return np.array(nb)


def siesta_ion_xml(fname):
    """
    Read the ion.xml file of a specie
    Input parameters:
    -----------------
    fname (str): name of the ion file
    Output Parameters:
    ------------------
    ion (dict): The ion dictionnary contains all the data
        from the ion file. Each field of the xml file give 
        one key. 
        The different keys are:
            'lmax_basis': int 
            'self_energy': float
            'z': int
            'symbol': str
            'label': str
            'mass': flaot
            'lmax_projs': int
            'basis_specs': str
            'norbs_nl': int
            'valence': float
            'nprojs_nl: int

            The following keys give the pao field,
            'npts': list of int
            'delta':list of float
            'cutoff': list of float
            'data':list of np.arrayof shape (npts[i], 2)
            'orbital': list of dictionnary
            'projector': list of dictionnary

    """
    doc = minidom.parse(fname)


    #the elements from the header
    elements_headers = [['symbol', str], ['label', str], ['z', int], ['valence', float], 
            ['mass', float], ['self_energy', float], ['lmax_basis', int], ['norbs_nl', int], 
            ['lmax_projs', int], ['nprojs_nl', int]]

    ion = {}
    for i, elname in enumerate(elements_headers):
        name = doc.getElementsByTagName(elname[0])
        ion[elname[0]] = get_data_elements(name[0], elname[1])

    #extract the basis_specs
    name = doc.getElementsByTagName("basis_specs")
    ion["basis_specs"] = getNodeText(name[0])

    extract_pao_elements(ion, doc)
    return ion

def getNodeText(node):
    nodelist = node.childNodes
    result = []
    for node in nodelist:
        if node.nodeType == node.TEXT_NODE:
            result.append(node.data)
    return ''.join(result)

def get_data_elements(name, dtype):
    """
    return the right type of the element value
    """
    if dtype is int:
        data = str2int(getNodeText(name))
        if len(data) > 1:
            return np.array(data)
        elif len(data) == 1:
            return data[0]
        else:
            raise ValueError("len(data)<1 ??")
    elif dtype is float:
        data = str2float(getNodeText(name))
        if len(data) > 1:
            return np.array(data)
        elif len(data) == 1:
            return data[0]
        else:
            raise ValueError("len(data)<1 ??")
    elif dtype is str:
        return getNodeText(name)
    else:
        raise ValueError('not implemented')

def extract_pao_elements(ion, doc):
    """
    extract the different pao element of the xml file
    Input Parameters:
    -----------------
        ion (dict)
        doc (minidom.parse)
    Output Parameters:
    ------------------
        ion (dict): the following keys are added to the ion dict:
            npts
            delta
            cutoff
            data
            orbital
            projector
    """

    name_npts = doc.getElementsByTagName("npts")
    name_delta = doc.getElementsByTagName("delta")
    name_cutoff = doc.getElementsByTagName("cutoff")
    name_data = doc.getElementsByTagName("data")

    name_orbital = doc.getElementsByTagName("orbital")
    name_projector = doc.getElementsByTagName("projector")

    ion["orbital"] = []
    ion["projector"] = []
    for i in range(len(name_orbital)):
        ion["orbital"].append(extract_orbital(name_orbital[i]))
    for i in range(len(name_projector)):
        ion["projector"].append(extract_projector(name_projector[i]))

    if len(name_data) != len(name_npts):
        raise ValueError("len(name_data) != len(name_npts): {0} != {1}".format(len(name_data), len(name_npts)))
    if len(name_data) != len(name_cutoff):
        raise ValueError("len(name_data) != len(name_cutoff): {0} != {1}".format(len(name_data), len(name_cutoff)))
    if len(name_data) != len(name_delta):
        raise ValueError("len(name_data) != len(name_delta): {0} != {1}".format(len(name_data), len(name_delta)))

    ion["npts"] = np.zeros((len(name_npts)), dtype=int)
    ion["delta"] = np.zeros((len(name_delta)), dtype=float)
    ion["cutoff"] = np.zeros((len(name_cutoff)), dtype=float)
    ion["data"] = []

    for i in range(len(name_data)):
        ion["npts"][i] = get_data_elements(name_npts[i], int)
        ion["cutoff"][i] = get_data_elements(name_cutoff[i], float)
        ion["delta"][i] = get_data_elements(name_delta[i], float)
        ion["data"].append(get_data_elements(name_data[i], float).reshape(ion["npts"][i], 2))

def extract_orbital(orb_xml):
    """
    extract the orbital
    """
    orb = {}
    #print('key: ', orb_xml.attributes.keys)
    #print('l values', orb_xml.attributes['l'].value)
    orb['l'] = int(orb_xml.attributes['l'].value)
    orb['n'] = int(orb_xml.attributes['n'].value)
    orb['z'] = int(orb_xml.attributes['z'].value)
    orb['ispol'] = int(orb_xml.attributes['ispol'].value)
    orb['population'] = float(orb_xml.attributes['population'].value)

    return orb

def extract_projector(pro_xml):
    """
    extract the projector
    """
    pro = {}
    pro['l'] = int(pro_xml.attributes['l'].value)
    pro['n'] = int(pro_xml.attributes['n'].value)
    #print('ref_energy', pro_xml.attributes['ref_energy'].value)
    pro['ref_energy'] = float(pro_xml.attributes['ref_energy'].value)

    return pro
