import { useState } from 'react';
import useAuth from '../components/auth/useAuth';
import * as api from '../components/auth/api'; 
import { toast } from 'sonner';
/**
 * custom hook to post ratings to the api
 * @author Louis Figes
 * @generated GitHub Copilot was used in the creation of this code.
 */
const useRatings = () => {
    const { accessToken } = useAuth();
    const [sending, setSending] = useState(true);
    const [response, setResponse] = useState([]);

    async function addRating(rating, movie_slug) {
        const formData = new FormData();
        formData.append('rating', rating);
        formData.append('movie_slug', movie_slug);
        setSending(true);
        try {
            const headers = { Authorization: `Bearer ${accessToken}` };
            const response = await api.post('ratings/create', formData, headers);
            if (response.success) {
                setResponse(response.data);
                toast.success('Rating added');
                return true;
            } else {
                toast.error(response.data.message);
            }
        } catch (error) {
            toast.error(response.data.message);
        }
        setSending(false);
        return false;
    }

    return { 
        sending, 
        response, 
        addRating
    };
};

export default useRatings;
