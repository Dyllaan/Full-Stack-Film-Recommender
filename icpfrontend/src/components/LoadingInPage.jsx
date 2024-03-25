import { SquareLoader } from 'react-spinners';

export default function LoadingInPage() {
    return (
      <div className="m-2">
        <SquareLoader loading={true} color={'#687387'} />
      </div>
    );
}