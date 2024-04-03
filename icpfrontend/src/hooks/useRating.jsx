import { useState } from 'react';
import useAuth from '../components/auth/useAuth';
import * as api from '../components/auth/api'; 
import { toast } from 'react-toastify';
/**
 * custom hook to check data from the api
 * just makes simple get requests easier
 * @author Louis Figes
 * @generated GitHub Copilot was used in the creation of this code.
 */
const useRating = () => {
    const { accessToken, loading : userLoading } = useAuth();
    const [loading, setLoading] = useState(true);

    async function sendRating() {
        if(userLoading) {
            return;
        }
        try {
            const response = await api.post('ratings/create', { Authorization: `Bearer ${accessToken}` });
            if(response.success) {
                return response.data;
            } else {
                toast.error("Your rating could not be saved.");
            }
        } catch (error) {
            toast.error("Your rating could not be saved.");
        } finally {
            setLoading(false);
        }
    }

    return { 
        sendRating,
        loading
    };
};

export default useRating;
