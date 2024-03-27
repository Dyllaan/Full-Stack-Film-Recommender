import { useState, useEffect } from 'react';
import Error from '../components/Message';
import useAuth from '../components/auth/useAuth';
import * as api from '../components/auth/api'; 
/**
 * custom hook to check data from the api
 * just makes simple get requests easier
 * @author Louis Figes
 * @generated GitHub Copilot was used in the creation of this code.
 */
const useFetchData = (initialEndpoint) => {
    const { accessToken, loading : userLoading } = useAuth();
    const [endpoint, setEndpoint] = useState(initialEndpoint);
    const [loading, setLoading] = useState(true);
    const [data, setData] = useState([]);
    const [error, setError] = useState(null);

    useEffect(() => {
        if (loading && !userLoading) {
            fetchData();
        }
    }, [endpoint, loading, userLoading]);

    async function fetchData() {
        try {
            const headers = accessToken ? { Authorization: `Bearer ${accessToken}` } : {};
            const response = await api.get(endpoint, headers);
            if (response.success) {
                setData(response.data);
            } else {
                throw new Error('No data found');
            }
        } catch (error) {
            setError(error.message);
        } finally {
            setLoading(false);
        }
    }

    function showInfo() {
        if (error) {
            return showError();
        }
    }

    function showError() {
        return <Error error={error} />;
    }

    return { 
        loading, 
        data, 
        error, 
        reloadData: () => setLoading(true), 
        showInfo,
        setEndpoint
    };
};

export default useFetchData;
