/**
 * Make HTTP request to API
 * @author Louis Figes
 * @generated Github Copilot was used in the creation of this code
 */
const BASE_URL = 'http://127.0.0.1:8000/';

async function handleResponse(response) {
    const contentType = response.headers.get("Content-Type");

    let responseBody;
    if (contentType && contentType.includes("application/json")) {
        responseBody = await response.json();
    } else {
        responseBody = await response.text();
    }

    const standardizedResponse = {
        success: response.ok,
        status: response.status,
        statusText: response.statusText,
        data: responseBody,
    };

    if (!response.ok) {
        standardizedResponse.error = responseBody.message || "An error occurred";
    }

    return standardizedResponse;
}

async function fetchWithTimeout(resource, options = {}, timeout = 8000) {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeout);
    try {
        const response = await fetch(resource, {
            ...options,
            signal: controller.signal  
        });
        clearTimeout(id);
        return response;
    } catch (error) {
        clearTimeout(id);
        throw error;
    }
}

export async function get(endpoint, headers = {}) {
    const response = await fetchWithTimeout(`${BASE_URL}${endpoint}`, {
        method: 'GET',
        headers: {
            ...headers
        }
    });
    return handleResponse(response);
}

export async function post(endpoint, data, headers = {}) {
    return httpRequest('POST', endpoint, data, headers);
}

export async function put(endpoint, data, headers = {}) {
    return httpRequest('PUT', endpoint, data, headers);
}

export async function del(endpoint, data, headers = {}) {
    return httpRequest('DELETE', endpoint, data, headers);
}

async function httpRequest(method, endpoint, data, headers = {}) {
    const response = await fetchWithTimeout(`${BASE_URL}${endpoint}`, {
        method,
        headers: {
            ...headers
        },
        body: data ? data : undefined
    });
    return handleResponse(response);
}