/**
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

////////////////////////////////////////////////////////////////////////////////
//
//  Utility funcs to wrap up saving a surface or the back buffer as a PPM file
//  In addition, wraps up a threshold comparision of two PPMs.
//
//  These functions are designed to be used to implement an automated QA testing for SDK samples.
//
//  Author: Bryan Dudash
//  Email: sdkfeedback@nvidia.com
//
// Copyright (c) NVIDIA Corporation. All rights reserved.
////////////////////////////////////////////////////////////////////////////////

#include <helper_functions.h>
#include <rendercheck_d3d11.h>

HRESULT CheckRenderD3D11::ActiveRenderTargetToPPM(ID3D11Device *pDevice, const char *zFileName)
{
    ID3D11DeviceContext *pDeviceCtxt;
    pDevice->GetImmediateContext(&pDeviceCtxt);
    ID3D11RenderTargetView *pRTV = NULL;
    pDeviceCtxt->OMGetRenderTargets(1,&pRTV,NULL);

    ID3D11Resource *pSourceResource = NULL;
    pRTV->GetResource(&pSourceResource);

    return ResourceToPPM(pDevice,pSourceResource,zFileName);
}

HRESULT CheckRenderD3D11::ResourceToPPM(ID3D11Device *pDevice, ID3D11Resource *pResource, const char *zFileName)
{
    ID3D11DeviceContext *pDeviceCtxt;
    pDevice->GetImmediateContext(&pDeviceCtxt);
    D3D11_RESOURCE_DIMENSION rType;
    pResource->GetType(&rType);

    if (rType != D3D11_RESOURCE_DIMENSION_TEXTURE2D)
    {
        printf("SurfaceToPPM: pResource is not a 2D texture! Aborting...\n");
        return E_FAIL;
    }

    ID3D11Texture2D *pSourceTexture = (ID3D11Texture2D *)pResource;
    ID3D11Texture2D *pTargetTexture = NULL;

    D3D11_TEXTURE2D_DESC desc;
    pSourceTexture->GetDesc(&desc);
    desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.Usage = D3D11_USAGE_STAGING;

    if (FAILED(pDevice->CreateTexture2D(&desc,NULL,&pTargetTexture)))
    {
        printf("SurfaceToPPM: Unable to create target Texture resoruce! Aborting... \n");
        return E_FAIL;
    }

    pDeviceCtxt->CopyResource(pTargetTexture,pSourceTexture);

    D3D11_MAPPED_SUBRESOURCE mappedTex2D;
    pDeviceCtxt->Map(pTargetTexture, 0, D3D11_MAP_READ,0,&mappedTex2D);

    // Need to convert from dx pitch to pitch=width
    unsigned char *pPPMData = new unsigned char[desc.Width*desc.Height*4];

    for (unsigned int iHeight = 0; iHeight<desc.Height; iHeight++)
    {
        memcpy(&(pPPMData[iHeight*desc.Width*4]),(unsigned char *)(mappedTex2D.pData)+iHeight*mappedTex2D.RowPitch,desc.Width*4);
    }

    pDeviceCtxt->Unmap(pTargetTexture, 0);

    // Prepends the PPM header info and bumps byte data afterwards
    sdkSavePPM4ub(zFileName, pPPMData, desc.Width, desc.Height);

    delete [] pPPMData;
    pTargetTexture->Release();

    return S_OK;
}

bool CheckRenderD3D11::PPMvsPPM(const char *src_file, const char *ref_file, const char *exec_path,
                                const float epsilon, const float threshold)
{
    char *ref_file_path = sdkFindFilePath(ref_file, exec_path);

    if (ref_file_path == NULL)
    {
        printf("CheckRenderD3D11::PPMvsPPM unable to find <%s> in <%s> Aborting comparison!\n", ref_file, exec_path);
        printf(">>> Check info.xml and [project//data] folder <%s> <<<\n", ref_file);
        printf("Aborting comparison!\n");
        printf("  FAILURE!\n");
        return false;
    }

    return sdkComparePPM(src_file,ref_file_path,epsilon,threshold,true) == true;
}