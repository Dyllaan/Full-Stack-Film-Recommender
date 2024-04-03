import { useState, useEffect } from 'react';
import Error from '../components/Message';
import useAuth from '../components/auth/useAuth';
import * as api from '../components/auth/api'; 
import { toast } from 'react-toastify';
/**
 * custom hook to check data from the api
 * just makes simple get requests easier
 * @author Louis Figes
 * @generated GitHub Copilot was used in the creation of this code.
 */
const useFetchData = (initialEndpoint, initialRun = true) => {
    const { accessToken, loading : userLoading, signOut } = useAuth();
    const [endpoint, setEndpoint] = useState(initialEndpoint);
    const [loading, setLoading] = useState(true);
    const [data, setData] = useState([]);
    const [error, setError] = useState(null);
    const [run, setRun] = useState(initialRun);

    useEffect(() => {
        if (loading && !userLoading && run) {
            fetchData();
        }
    }, [endpoint, loading, userLoading, run]);

    function doRun() {
        setRun(true);
    }

    async function fetchData() {
        try {
            const headers = accessToken ? { Authorization: `Bearer ${accessToken}` } : {};
            const response = await api.get(endpoint, headers);
            if (response.success) {
                setData(response.data);
            } else {
                if(response.status == 401) {
                    toast.error("You have been logged out, please log back in to continue");
                    signOut();
                }
                toast.error(response.error);
            }
        } catch (error) {
            setError(error.message);
        } finally {
            setLoading(false);
            if(!initialRun) {
                setRun(false);
            }
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
        setEndpoint,
        doRun
    };
};

export default useFetchData;
