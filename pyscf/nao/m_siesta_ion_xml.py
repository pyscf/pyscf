# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division, print_function
import numpy as np
from xml.dom import minidom
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

    #for node in doc.getElementsByTagName('paos'):  # visit every node <bar />
    #  #print node.toxml()
    #  for delt in node.getElementsByTagName('delta'):
    #    print getNodeText(delt)
    #node = doc.getElementsByTagName("pao")
    #print('pao: ', node)

    field = {'paos': 'orbital', 'kbs': 'projector', 'vna': None, 'chlocal': None, 'reduced_vlocal': None, 'core': None}
    for k, v in field.items():
      ion[k] = {}
      if (len(doc.getElementsByTagName(k))>0):
        extract_field_elements(ion[k], doc.getElementsByTagName(k)[0], field=v)
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

def extract_field_elements(pao, doc, field=None):
    """
    extract the different pao element of the xml file
    Input Parameters:
    -----------------
        pao (dict): dict containing the pao element
        field: field name of the node
        doc (minidom.parse)
    Output Parameters:
    ------------------
        ion (dict): the following keys are added to the ion dict:
            npts
            delta
            cutoff
            data
            orbital
    """

    pao['npaos'] = len(doc.getElementsByTagName('delta'))
    if pao['npaos'] != len(doc.getElementsByTagName('cutoff')) or\
      pao['npaos'] != len(doc.getElementsByTagName('npts')):
      raise ValueError('Error reqding ion file! npaos is not constant??')
    pao['delta'] = np.zeros((pao['npaos']), dtype=float)
    pao['cutoff'] = np.zeros((pao['npaos']), dtype=float)
    pao['npts'] = np.zeros((pao['npaos']), dtype=int)

    pao['data'] = []
    for i, delt in enumerate(doc.getElementsByTagName('delta')):
      pao['delta'][i] = get_data_elements(delt, float)
    for i, delt in enumerate(doc.getElementsByTagName('cutoff')):
      pao['cutoff'][i] = get_data_elements(delt, float)
    for i, delt in enumerate(doc.getElementsByTagName('npts')):
      pao['npts'][i] = get_data_elements(delt, int)

    for i, dat in enumerate(doc.getElementsByTagName('data')):
      pao['data'].append(get_data_elements(dat, float).reshape(pao["npts"][i], 2))

    if len(pao['data']) != pao['npaos']:
      raise ValueError('Error reading ion file, len(data) != npaos')



    if field is not None:
      name_orbital = doc.getElementsByTagName(field)
      pao[field] = []

      if field == 'orbital':
        for i in range(len(name_orbital)):
          pao[field].append(extract_orbital(name_orbital[i]))
      elif field == 'projector':
        for i in range(len(name_orbital)):
          pao[field].append(extract_projector(name_orbital[i]))
      else:
        raise ValueError(field + ' not implemented, onlt orbital or projector!!')
      
      if len(pao[field]) != pao['npaos']:
        raise ValueError('Error reading ion file, len(' + field +') != npaos')

    #for k, val in pao.items():
    #  print(k + ': ', val)

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

#
# Executable part
#
if __name__=="__main__":
  import sys 
  fname = sys.argv[1]
  ionxml = siesta_ion_xml(fname)
  print(dir(ionxml))
  

